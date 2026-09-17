import json
from pathlib import Path
from token_yield.customer_models import (
    fit_models,
    predict_cost,
    evaluate_predictions
)
from token_yield.robust import Record, RidgeLinearModel
from token_yield.customer_decomposition import SCOPING_BRICKS
import matplotlib.pyplot as plt

def token_record(row):
    quote = row["quote"]
    features = (
        quote["prompt_bytes"],
        quote["planned_output_tokens"],
        *(quote["counts"][op] for op in SCOPING_BRICKS),
    )
    return Record(
        features=features,
        target=row["usage"]["total_tokens"],
        group=row["project_id"],
    )

if __name__ == "__main__":
    # Path to the JSON file containing the customer scoping records.
    path = Path(r"runs/20260917_customer_scoping_fresh_holdout_v1/records.json");
    # Load the records from the JSON file in memory.
    records = json.loads(path.read_text(encoding="utf-8"))
    
    # Training and test sets, unprocessed.
    train_raw = [x for x in records if x["split"] == "train"]
    test_raw = [x for x in records if x["split"] == "holdout"]
    
    # Ensure the sets are disjoint.
    assert not any(x in test_raw for x in train_raw), "Training and test sets are not disjoint."

    # Convert raw records to token records for training and testing.
    train = [token_record(r) for r in train_raw]
    test = [token_record(r) for r in test_raw]

    # Train the model.
    model = RidgeLinearModel.fit(train, alpha=10.0)

    # Evaluate the model.
    errors = [
        abs(max(0.0, model.predict(r.features)) - r.target)
        for r in test
    ]
    print("Holdout MAE (tokens):", sum(errors) / len(errors))

    # Plot the results - predicted vs actual tokens.
    actual = [r.target for r in test]
    predicted = [max(0.0, model.predict(r.features)) for r in test]

    plt.scatter(actual, predicted)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')  # Diagonal line for reference
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Actual Tokens")
    plt.ylabel("Predicted Tokens")
    plt.title("Predicted vs Actual Tokens")
    plt.show()