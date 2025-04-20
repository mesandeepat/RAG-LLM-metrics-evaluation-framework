import os

import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness

from utils import load_test_data, get_llm_response

os.environ["RAGAS_APP_TOKEN"] = ""
@pytest.mark.asyncio
@pytest.mark.parametrize("getdata",
                         load_test_data("dualmetrics.json"), indirect=True
                         )

async def test_relevancy_factual(llm_wrapper, getdata):
    metrics = [ResponseRelevancy(llm=llm_wrapper),
               FactualCorrectness(llm=llm_wrapper)]

    eval_dataset = EvaluationDataset([getdata])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    print(results)
    results.upload()


@pytest.fixture()
def getdata(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        reference=test_data["reference"]
    )

    return sample
