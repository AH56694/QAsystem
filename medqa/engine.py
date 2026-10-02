import json
from dataclasses import dataclass,asdict,field


from .schema import INTENTS

@dataclass
class Prepared:
    entities: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    note :str = ""


class Engine:
    def __init__(self,recognizer,graph,llm):
        self.recognizer ,self.graph,self.llm = recognizer,graph,llm

    def prepare(self,question):
        question = question.strip()
        if not question or len(question) == 300:
            raise ValueError("请输入1-300字符的问题")
        entities = self.recognizer.find(question)
        result = Prepared(entities=[asdict(e)for e in entities])
        if any(not e.canonical for e in entities):
            result.note = "有名称无法可靠匹配，请使用资料中的完整疾病或药品名称重试"
            return result
        diseases=[e.canonical for e in entities if e.kind == "疾病"]
        symptoms=[e.canonical for e in entities if e.kind == "疾病症状"]
        if symptoms and not diseases and not any(e.kind == "药品" for e in entities):
            candidates= self.graph.symptom_candidates(symptoms)
            names = "、".join(row["name"] for row in candidates)
            result.note = (f"资料中与这些症状有关联的疾病包括：{names or '未找到'}。"
                       "这不是诊断结果。请明确想查询的疾病名称，或咨询医生。")
            return result
        result.intents = self.llm.classify(question)
        if not result.intents:
            result.note = "当前支持疾病资料和药品生产商查询，请提出具体问题。"
            return result
        for intent in result.intents:
            spec = INTENTS[intent]
            names = list(dict.fromkeys(e.canonical for e in entities if e.kind == spec.entity_type))
            if not names:
                result.missing.append(f"{intent}：缺少明确的{spec.entity_type}名称")
            for name in names:
                values = self.graph.query(intent,name)
                if not values:
                    result.missing.append(f"{name} / {intent}：资料未收录")
                for value in values:
                    result.evidence.append({"id": f"E{len(result.evidence) + 1}",
                                        "subject": name, "intent": intent,
                                        "predicate": spec.key, "value": value[:2000],
                                        "source": "本项目导入的 medical_new_2.json"})

        if not result.evidence:
            result.note = "没有专业的知识库证据"+";".join(result.missing)
        return result


    def answer(self,question,prepared):
        if prepared.note:
            yield prepared.note
            return
        selected,size = [],0
        for item in prepared.evidence:
            cost = len(json.dumps(item,ensure_ascii=False))
            if size +cost >12000:
                break
            selected.append(item)
            size+=cost
        yield from self.llm.generate(question,selected)
        if prepared.missing:
            yield "\n\n未完成的查询："+";".join(prepared.missing)
        if len(selected)<len(prepared.evidence):
            yield "\n\n本次生成仅使用部分证据；完整检索结果见证据面板"

def create_engine(settings):
    from .entities import EntityRecognizer
    from .graph import KnowledgeGraph,connect
    from .llm import OllamaModel
    return Engine(EntityRecognizer.from_settings(settings),KnowledgeGraph(connect(settings),settings.project),OllamaModel(settings))
