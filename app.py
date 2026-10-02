import logging
import time
from dataclasses import asdict, replace
from uuid import uuid4

import streamlit as st

from medqa.auth import Accounts
from medqa.config import load_settings
from medqa.engine import create_engine
from medqa.graph import KnowledgeUnavailable
from medqa.llm import ModelUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="我的医疗资料问答", page_icon="📚")
settings = load_settings()
accounts = Accounts(settings.accounts)


@st.cache_resource
def cached_engine(config):
    return create_engine(config)


def login_page():
    st.title("我的医疗资料问答")
    with st.form("account"):
        action = st.radio("操作", ["登录", "注册"])
        name = st.text_input("用户名")
        password = st.text_input("密码", type="password", max_chars=128)
        submitted = st.form_submit_button("提交")
    if not submitted:
        return
    if time.time() < st.session_state.get("blocked_until", 0):
        st.error("尝试过于频繁，请一分钟后再试")
        return
    try:
        if action == "注册":
            accounts.create(name, password)
            st.success("注册成功，请切换到登录")
        else:
            user = accounts.authenticate(name, password)
            if user is None:
                count = st.session_state.get("failures", 0) + 1
                st.session_state.failures = count
                if count >= 5:
                    st.session_state.blocked_until = time.time() + 60
                    st.session_state.failures = 0
                st.error("用户名或密码错误")
            else:
                st.session_state.clear()
                st.session_state.user = user
                st.rerun()
    except ValueError as exc:
        st.error(str(exc))


if "user" not in st.session_state:
    login_page()
    st.stop()

user = st.session_state.user
st.sidebar.write(f"当前用户：{user['name']}")
if st.sidebar.button("退出登录"):
    st.session_state.clear()
    st.rerun()
if "rooms" not in st.session_state:
    st.session_state.rooms = {"对话 1": []}
if st.sidebar.button("新建对话"):
    name = f"对话 {len(st.session_state.rooms) + 1}"
    st.session_state.rooms[name] = []
    st.session_state.active_room = name
room = st.sidebar.selectbox("对话窗口", list(st.session_state.rooms), key="active_room")
messages = st.session_state.rooms[room]
if st.sidebar.button("清空当前对话"):
    messages.clear()
    st.rerun()
model = st.sidebar.selectbox("回答模型", [settings.llm, "qwen:1.8b"])
st.title("医疗知识图谱问答")
st.caption("学习数据资料查询。每次请写出完整疾病或药品名称；历史记录仅用于展示。")
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("evidence"):
            with st.expander("查看知识库证据"):
                st.json(message["evidence"])
question = st.chat_input("例如：高血压应该挂什么科？", max_chars=300)
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    trace = uuid4().hex[:8]
    started = time.perf_counter()
    with st.chat_message("assistant"):
        output = st.empty()
        try:
            with st.spinner("正在识别问题并检索资料……"):
                engine = cached_engine(replace(settings, llm=model))
                prepared = engine.prepare(question)
            answer = ""
            for token in engine.answer(question, prepared):
                answer += token
                output.markdown(answer + " ▌")
            output.markdown(answer)
            messages.append({"role": "assistant", "content": answer, "evidence": prepared.evidence})
            if prepared.evidence:
                with st.expander("查看知识库证据"):
                    st.json(prepared.evidence)
            if user["role"] == "admin":
                with st.expander("管理员调试信息"):
                    st.json(asdict(prepared))
            logging.info("request=%s status=ok seconds=%.2f", trace, time.perf_counter() - started)
        except (ValueError, FileNotFoundError, KnowledgeUnavailable, ModelUnavailable) as exc:
            output.error(str(exc))
            messages.append({"role": "assistant", "content": str(exc)})
            logging.warning("request=%s error=%s", trace, type(exc).__name__)
        except Exception as exc:
            output.error(f"应用发生错误，请管理员检查配置。请求编号：{trace}")
            logging.error("request=%s error=%s", trace, type(exc).__name__)