from dataclasses import dataclass


TYPES = ("疾病", "药品", "食物", "检查项目", "科目", "疾病症状", "治疗方法", "药品商")
ATTRS = {
    "疾病简介": "desc", "疾病病因": "cause", "预防措施": "prevent",
    "治疗周期": "cure_lasttime", "治愈概率": "cured_prob", "疾病易感人群": "easy_get",}
REL_FIELDS={
    "疾病使用药品": ("药品", ("common_drug", "recommand_drug")),
    "疾病宜吃食物": ("食物", ("do_eat", "recommand_eat")),
    "疾病忌吃食物": ("食物", ("not_eat",)),
    "疾病所需检查": ("检查项目", ("check",)),
    "疾病所属科目": ("科目", ("cure_department",)),
    "疾病的症状": ("疾病症状", ("symptom",)),
    "治疗的方法": ("治疗方法", ("cure_way",)),
    "疾病并发疾病": ("疾病", ("acompany",)),
}
RELATIONS=(*REL_FIELDS,"生产")

@dataclass(frozen=True)
class QuerySpec:
    kind: str
    key: str
    target: str = ""
    entity_type: str = "疾病"

INTENTS= {
    "查询疾病简介": QuerySpec("attribute", "疾病简介"),
    "查询疾病病因": QuerySpec("attribute", "疾病病因"),
    "查询疾病预防措施": QuerySpec("attribute", "预防措施"),
    "查询疾病治疗周期": QuerySpec("attribute", "治疗周期"),
    "查询治愈概率": QuerySpec("attribute", "治愈概率"),
    "查询疾病易感人群": QuerySpec("attribute", "疾病易感人群"),
    "查询疾病所需药品": QuerySpec("relation", "疾病使用药品", "药品"),
    "查询疾病宜吃食物": QuerySpec("relation", "疾病宜吃食物", "食物"),
    "查询疾病忌吃食物": QuerySpec("relation", "疾病忌吃食物", "食物"),
    "查询疾病所需检查项目": QuerySpec("relation", "疾病所需检查", "检查项目"),
    "查询疾病所属科目": QuerySpec("relation", "疾病所属科目", "科目"),
    "查询疾病的症状": QuerySpec("relation", "疾病的症状", "疾病症状"),
    "查询疾病的治疗方法": QuerySpec("relation", "治疗的方法", "治疗方法"),
    "查询疾病的并发疾病": QuerySpec("relation", "疾病并发疾病", "疾病"),
    "查询药品的生产商": QuerySpec("reverse", "生产", "药品商", "药品"),
}
"""- TYPES 规定八种节点标签，疾病排在前面也定义了词典歧义时的默认优先级。
- ATTRS 把中文图谱属性名映射到 JSON 字段。疾病简介是属性，没有另建一个“疾病简介”节点。
- REL_FIELDS 声明“读哪些字段，连到哪种节点”。例如 common_drug 与 recommand_drug 合并为同一种关系。生产商的字符串需要额外拆分，所以在导入模块单独处理。
- QuerySpec 是一条查询的说明书。kind 为 attribute 时查属性；relation 查从疾病出发的边；reverse 从药品沿反向边查厂商。
- INTENTS 的键是允许 LLM 返回的完整名称。LLM 不能自己发明数据库字段，也不能返回一段 Cypher 要求程序执行。
"""
