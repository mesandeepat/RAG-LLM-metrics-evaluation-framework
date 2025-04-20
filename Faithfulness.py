import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, faithfulness

from utils import get_llm_response, load_test_data

@pytest.mark.parametrize("getdata", load_test_data("faithfulness.json"), indirect=True)
@pytest.mark.asyncio
async def test_faithfulness(llm_wrapper, getdata):
    faithful = Faithfulness(llm=llm_wrapper)
    score = await faithful.single_turn_ascore(getdata)
    print(score)
    assert score > 0.8


@pytest.fixture()
def getdata(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"]]
    )
    return sample
