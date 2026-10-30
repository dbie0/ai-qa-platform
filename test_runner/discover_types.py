import os
import json
import argparse
import pymongo


def get_test_plan_data(target_test_name):
    # Fetch MongoDB URL from environment variable
    mongo_url = os.environ.get("MONGO_URI")
    if not mongo_url:
        raise EnvironmentError("Error: MONGO_URL environment variable is not set.")

    client = pymongo.MongoClient(mongo_url)
    db = client["validation"]
    collection = db["test_definitions"]

    plan = collection.find_one({"test_name": target_test_name})

    if not plan:
        return {
            "ml_toolkit": None,
            "unique_tools": []
        }

    unique_tools = set()
    ml_toolkit = plan.get("ml_toolkit")

    for config in plan.get("configurations", []):
        for test in config.get("tests", []):
            test_name_str = test.get("test_name", "")
            tool, _ = test_name_str.split(".")
            unique_tools.add(tool)

    return {
        "ml_toolkit": ml_toolkit,
        "unique_tools": list(unique_tools)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch test plan tools and ML toolkit from MongoDB.")
    parser.add_argument("test_name", type=str,
                        help="The name of the test plan to query (e.g., Qwen2.5-1.5B-Instruct_Basic)")

    args = parser.parse_args()

    result = get_test_plan_data(args.test_name)

    # Print valid JSON to stdout so Jenkins can parse it
    print(json.dumps(result))