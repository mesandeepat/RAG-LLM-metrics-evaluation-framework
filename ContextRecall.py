import os

import pytest
import requests
from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall

from utils import get_llm_response, load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize("getdata", load_test_data("contextRecall.json"), indirect=True
                         )
async def test_context_recall(llm_wrapper, getdata):

    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(getdata)
    print(score)
    assert score > 0.7

@pytest.fixture()
def getdata(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)


    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        reference=test_data["reference"]
    )
    return sample