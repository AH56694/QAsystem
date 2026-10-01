from sys import prefix

from py2neo import Graph

from .schema import INTENTS

class KnowledgeUnavailable(RuntimeError):
    pass


def connect(settings):
    from py2neo import Graph
    if not settings.neo4j_password:
        raise KnowledgeUnavailable("请设置正确的neo4j密码")
    try:
        graph = Graph(settings.neo4j_url, auth=(settings.neo4j_user,settings.neo4j_password),name=settings.neo4j_database)
        graph.run("RETURN 1 AS ok ").data()
        return graph
    except Exception as exc:
        raise KnowledgeUnavailable("无法连接知识库，请检查数据库服务与链接配置") from exc


class KnowledgeGraph:
    def __init__(self,graph,project):
        self.graph,self.project = graph,project

    def query(self ,intent,name):
        spec = INTENTS[intent]
        prefix= f"MATCH(a:{spec.entity_type}{{项目:$project,名称:$name}})"
        if spec.kind == "relation":
            cypher = prefix + f"MATCH (a)-[:`{spec.key}`]->(b:`{spec.target}`) "
            cypher += "WHERE b.项目=$project RETURN DISTINCT b.名称 AS value ORDER BY value LIMIT 30"
        else:
            cypher = prefix + f"MATCH (a)<-[:`{spec.key}`]-(b:`{spec.target}`) "
            cypher += "WHERE b.项目=$project RETURN DISTINCT b.名称 AS value ORDER BY value LIMIT 30"
        try:
            rows = self.graph.run(cypher,project=self.project,name=name,key=spec.key).data()
        except Exception as exc:
            raise KnowledgeUnavailable("知识库查询失败，请稍后重试或检查数据库") from exc
        return [str(row["value"]) for row in rows if row.get("value")]
    def symptom_candidates(self,symptoms):
        try:
            return self.graph.run(
                "MATCH (d:`疾病` {项目:$project})-[:`疾病的症状`]->(s:`疾病症状`) "
                "WHERE s.项目=$project AND s.名称 IN $symptoms "
                "RETURN d.名称 AS name, count(DISTINCT s) AS hits "
                "ORDER BY hits DESC, name LIMIT 5",
                project=self.project,symptoms=symptoms
            ).data()
        except Exception as exc:
            raise KnowledgeUnavailable("症状候选查询失败") from exc

