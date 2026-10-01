import json
from dataclasses import dataclass,replace

import ahocorasick
from markdown_it.common.entities import entities
from numpy.f2py.crackfortran import kindselector
from sklearn.feature_extraction.text import TfidfVectorizer

from.schema import TYPES


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    kind: str
    text: str
    canonical: str = ""
    source: str = "rule"
    score: float = 1.0

def non_overlapping(entities):
    selected,occupied =[],set()
    ordered=sorted(entities,key=lambda e:(
            -(e.end-e.start),e.source !="rule",e.start,TYPES.index(e.kind)))
    for item in ordered:
        positions = set(range(item.start,item.end))
        if positions and not positions &occupied:
            selected.append(item)
            occupied.update(positions)
    return sorted(selected,key=lambda e: e.start)

class RuleMatcher:
    def __init__(self,lexicon):
        self.automaton = ahocorasick.Automaton()
        words = {}
        for kind in TYPES:
            for word in lexicon.get(kind,[]):
                if word :
                    words.setdefault(word,[]).append(kind)
        if not words:
            raise ValueError("词典为空，请先生成词典")
        for word,kinds in words.items():
            self.automaton.add_word(word,(word,kinds))
        self.automaton.make_automaton()

    def find(self,text):
        result = []
        for end,(word,kinds) in self.automaton.iter(text):
            for kind in kinds:
                result.append(Entity(end+1-len(word),end+1,kind,word,word))
        return non_overlapping(result)


class ALigner:
    """只接受高相似且领先次优候选的结果；不会强行对齐每个词."""
    def __init__(self,lexicon):
        self.index={}
        for kind,names in lexicon.items():
            names=sorted(set(names))
            if not names:
                continue
            vectorizer = TfidfVectorizer(analyzer="char",ngram_range=(1,2))
            matrix = vectorizer.fit_transform(names)
            self.index[kind]= (names,set(names),vectorizer,matrix)

    def align(self,entity):
        if entity.kind not in self.index:
            return replace(entity,canonical="",score=0)
        names,exact,vectorizer,matrix = self.index[entity.kind]
        if entity.text in exact:
            return replace(entity,canonical=entity.text,score=0.0)
        scores = (matrix @ vectorizer.transform([entity.text]).T).tocoo()
        ranked = sorted(zip(scores.row,scores.data),key=lambda row:-row[1])
        if not ranked:
            return replace(entity,canonical="",score=0)
        index,best = ranked[0]
        second = ranked[1][1] if len(ranked)>1 else 0.0
        accepted = best >=0.85 and best -second>=0.10
        return replace(entity,canonical=names[index] if accepted else"",score= float(best))


class EntityRecognizer:
    def __init__(self,lexicon,predictor = None):
        self.rules = RuleMatcher(lexicon)
        self.aligner = ALigner(lexicon) if predictor is not None else None
        self.predictor = predictor

    @classmethod
    def from_settings(cls,settings):
        lexicon = json.loads(settings.lexicon.read_text(encoding="utf-8"))
        predictor = None
        if settings.entity_mode == "hybrid":
            from .ner import Predictor
            predictor = Predictor(settings.base_model,settings.checkpoint)
        return cls(lexicon,predictor)

    def find(self,text):
        entities = self.rules.find(text)
        if self.predictor is not None:
            learned= [self.aligner.align(e) for e in self.predictor.find(text)]
            entities = non_overlapping(entities+learned)
        return entities

