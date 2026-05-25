import pytest
from app.intent.jieba_classifier import JiebaIntentClassifier


def test_jieba_hit_legal():
    c = JiebaIntentClassifier(threshold=0.6)
    result = c.classify("招标投标法对投标保证金有什么规定？")
    assert result.hit is True
    assert "legal" in result.intents
    assert result.level == "L1"


def test_jieba_hit_tender():
    c = JiebaIntentClassifier(threshold=0.6)
    result = c.classify("查询北京地区最近的招标公告")
    assert result.hit is True
    assert "tender" in result.intents
    assert result.level == "L1"


def test_jieba_miss():
    c = JiebaIntentClassifier(threshold=0.6)
    result = c.classify("今天天气怎么样")
    assert result.hit is False
    assert result.level == "L1"