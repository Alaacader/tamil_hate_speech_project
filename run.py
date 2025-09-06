import argparse
from src.train_models import train_and_select_best
from src.evaluate import evaluate_saved_model
from src.predict import predict_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train models and save best pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved best pipeline")
    parser.add_argument("--predict", action="store_true", help="Quick predict on a text")
    parser.add_argument("--text", type=str, default=None, help="Text to predict (if --predict)")

    args = parser.parse_args()

    if args.train:
        best_name, results = train_and_select_best()
        print("\nBest model:", best_name)
        print("All results (name, CV_F1, val_acc, best_params):")
        for row in results:
            print(row)

    if args.evaluate:
        evaluate_saved_model()

    if args.predict:
        if not args.text:
            print("Please pass --text 'your sentence'")
        else:
            out = predict_texts([args.text])
            print(out)

if __name__ == "__main__":
    main()
