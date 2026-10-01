import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .config import load_settings
from .schema import ATTRS,REL_FIELDS,TYPES

def read_records(path):
    """同名疾病使用最后一条完整记录；格式错误时停止并报告行号"""
    records = {}
    duplicates = 0
    with Path(path).open(encoding="utf-8-sig") as source:
        for number,line in enumerate(source,1):
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip().rstrip(","))
                name = item["name"].strip()
                if not name:
                    raise ValueError("疾病名称为空")
            except(ValueError, KeyError, TypeError,AttributeError) as exc:
                raise ValueError(f"数据第{number}行格式错误") from exc
            duplicates += name in records
            records[name] = item
    if not records:
        raise ValueError("数据文件没有有效记录")
    return list(records.values()),duplicates


def strings(value):
    """<UNK>"""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [word for child in value for word in strings(child)]
    return []


def build_graph(records):
    nodes= {kind:{}for kind in TYPES}
    edges = set()

    def node(kind,name,**properties):
        nodes[kind].setdefault(name,{"名称":name}).update(properties)

    def edge(left_kind,left,relation,right_kind,right):
        node(left_kind,left)
        node(right_kind,right)
        edges.add((left_kind,left,relation,right_kind,right))

    for item in records:
        disease = item["name"].strip()
        props = {key:";".join(strings(item.get(field))) for key,field in ATTRS.items()}
        node("疾病",disease,**props)
        for relation ,(target,fields) in REL_FIELDS.items():
            values = [word for field in fields for word in strings(item.get(field))]
            if relation =="疾病所属科目":
                for word in values:
                    node(target,word)
                values = values[-1:]
            if relation =="疾病的症状":
                values= [word.removesuffix("...").strip() for word in values]
            for word in filter (None,values):
                edge("疾病",disease,relation,target,word)
        for detail in strings(item.get("drug_detail")):
            parts = detail.split(",")
            if len(parts) == 2 and all(p.strip() for p in parts):
                drug,company = (p.strip() for p in parts)
                edge("药品商",company,"生产","药品",drug)
    return nodes,sorted(edges)

def batches(rows,size= 500):
    for start in range(0,len(rows),size):
        yield rows[start:start+size]

def import_graph(graph,project,nodes,edges):
    """标识符来自build_graph中的常量；所有数据值都通过参数传递"""
    for index,(kind,values) in enumerate(nodes.items()):
        graph.run(f"CREATE CONSTRAINT medqa_{index} IF NOT EXISTS "
                  f"FOR (n:`{kind}`) REQUIRE (n.项目, n.名称) IS UNIQUE")
        for rows in batches(list(values.values())):
            graph.run(f"UNWIND $rows AS row MERGE (n:`{kind}` "
                      "{项目:$project, 名称:row.名称}) SET n += row",
                      rows= rows,project=project)

    groups= defaultdict(list)
    for left_kind,left,relation,right_kind,right in edges:
        groups[left_kind,relation,right_kind].append({"left":left,"right":right})
    for (left_kind,relation,right_kind),values in groups.items():
        for rows in batches(values):
            graph.run(f"UNWIND $rows AS row MATCH (a:`{left_kind}` "
                      "{项目:$project, 名称:row.left}) "
                      f"MATCH (b:`{right_kind}` {{项目:$project, 名称:row.right}}) "
                      f"MERGE (a)-[:`{relation}`]->(b)",rows= rows,project=project)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source",type = Path)
    parser.add_argument("--write",action = "store_true",help = "执行图谱写入")
    args = parser.parse_args()
    settings = load_settings()
    records,duplicates = read_records(args.source)
    nodes,edges = build_graph(records)
    settings.lexicon.parent.mkdir(parents=True,exist_ok=True)
    settings.lexicon.write_text(json.dumps({kind:sorted(values) for kind,values in nodes.items()},ensure_ascii=False),encoding="utf-8")
    report = {"records":len(records),"duplicates":duplicates,
              "nodes":{kind:len(values) for kind,values in nodes.items()},
              "edges":len(edges),"project":settings.project,
              "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest()}
    (settings.lexicon.parent/"import_report.json").write_text(json.dumps(report,ensure_ascii=False,indent= 2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.write:
        from .graph import connect
        import_graph(connect(settings),settings.project,nodes,edges)
        print("图谱写入完成；相同数据可以重复导入")


if __name__ == "__main__":
    main()