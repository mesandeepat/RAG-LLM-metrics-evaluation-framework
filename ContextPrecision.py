#pytest
import os

import pytest
from pyparsing import results
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference

from utils import get_llm_response, load_test_data
os.environ["RAGAS_APP_TOKEN"] = ""
@pytest.mark.asyncio
@pytest.mark.parametrize("getdata",
                         load_test_data("contextPrecision.json"), indirect=True
                         )
async def test_context_precision(llm_wrapper, getdata):

    #Step 1 - create object of class - its in conftest class
    context_precision = LLMContextPrecisionWithReference(llm=llm_wrapper)

    #Step 2 - Feed Data
    #moved to pytest.fixture
    #Step 3 - score
    score = await context_precision.single_turn_ascore(getdata)
    print(score)

    assert score > 0.8




@pytest.fixture()
def getdata(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    question = "How many articles are there in the Selenium webdriver python course?"

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        reference=test_data["reference"]
    )
    return sample
