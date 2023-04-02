import json
import csv
import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def read_test_file(test_file_path):
    with open(test_file_path, "r") as f:
        data = json.load(f)
    return data["data"]

def predict(model, tokenizer, input_text, max_length, temp, topk, topp, rep_pen, num_beams, do_sample, forced_bos_token_id):
    input_options = {"text": input_text}
    if forced_bos_token_id:
        input_options["forced_bos_token_id"] = int(forced_bos_token_id)

    inputs = tokenizer.encode(**input_options, return_tensors="pt")
    model_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }
    if temp:
        model_kwargs["temperature"] = temp
    if topk:
        model_kwargs["top_k"] = topk
    if topp:
        model_kwargs["top_p"] = topp
    if rep_pen:
        model_kwargs["repetition_penalty"] = rep_pen

    outputs = model.generate(inputs, **model_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the trained model")
    parser.add_argument("--test_file_path", help="Path to the test file")
    parser.add_argument('--count', type=int,
                        help='Number of predicted sentences. Defaults to no. of source sentences.')
    parser.add_argument('--max_length', type=int, help='Max length of predicted sentence. Defaults to 700.')
    parser.add_argument('--output', type=str, help='Output file name.', required=True)
    parser.add_argument('--verbosity', type=str,
                        help='Specify verbosity using h (High), m (Medium), l (Low). Defaults to m.')
    parser.add_argument('--temp', type=float, help='Temperature for random sampling.')
    parser.add_argument('--topk', type=int, help='Top k for random sampling.')
    parser.add_argument('--topp', type=float, help='Top p for random sampling.')
    parser.add_argument('--rep_pen', type=float, help='Repetition penalty.')
    parser.add_argument('--num_beams', type=int, help="Number of beams. Default is 1")
    parser.add_argument('--do_sample', type=bool, help="Whether to use sampling. Default False")
    parser.add_argument('--forced_bos_token_id', type=str, help="Forced BOS token to select task")

    args = parser.parse_args()

    logging.info("Starting predictions on %s." % 'cuda' if torch.cuda.is_available() else 'cpu. Unable to find cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set verbosity.
    verbose = 500
    if args.verbosity == "h":
        verbose = 100
    elif args.verbosity == "l":
        verbose = 1000

    logging.info("Verbosity set to %s." % args.verbosity)

    # Set max length
    max_length = args.max_length if args.max_length else 700
    logging.info("Max length set to %d." % max_length)

    temp = args.temp if args.temp else 1.0
    logging.info("Temperature set to %f." % temp)

    topk = args.topk if args.topk else 50
    logging.info("Top k set to %d." % topk)

    topp = args.topp if args.topp else 1.0
    logging.info("Top p set to %f." % topp)

    rep_pen = args.rep_pen if args.rep_pen else 1.0
    logging.info("Repetition penalty set to %f." % rep_pen)

    num_beams = args.num_beams if args.num_beams else 1
    logging.info("Number of beams is set to %d." % num_beams)

    do_sample = args.do_sample if args.do_sample else False
    logging.info("Do sampling is set to %s." % do_sample)

    model, tokenizer = load_model(args.model_path)



    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test_data = read_test_file(args.test_file_path)

    forced_bos_token_id = (
        tokenizer.lang_code_to_id[args.forced_bos_token_id] if args.forced_bos_token_id is not None else None
    )

    if args.forced_bos_token_id is not None:
        model.config.forced_bos_token_id = forced_bos_token_id
    model.to(device)

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["task_id", "input", "predicted", "expected"])

        for item in test_data:
            task_id = item["lang1"].split(":")[0]
            input_text = item["lang1"].strip()
            expected_text = item["lang2"].split(":")[1].strip()

            input_options = {"text": input_text}

            inputs = tokenizer(**input_options, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            model_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
            }
            if temp:
                model_kwargs["temperature"] = temp
            if topk:
                model_kwargs["top_k"] = topk
            if topp:
                model_kwargs["top_p"] = topp
            if rep_pen:
                model_kwargs["repetition_penalty"] = rep_pen

            output_ids = model.generate(input_ids=input_ids, **model_kwargs)
            predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            csv_writer.writerow([task_id, input_text, predicted_text, expected_text])

            if args.verbosity.lower() in {"h", "high"}:
                print(f"Task ID: {task_id}")
                print(f"Input: {input_text}")
                print(f"Predicted: {predicted_text}")
                print(f"Expected: {expected_text}")
                print("-" * 50)
    
            elif args.verbosity.lower() in {"m", "medium"}:
                print(f"Task ID: {task_id}")
                print(f"Input: {input_text}")
                print(f"Predicted: {predicted_text}")
                print("-" * 50)
    
            elif args.verbosity.lower() in {"l", "low"}:
                print(f"Task ID: {task_id}")
                print(f"Predicted: {predicted_text}")
                print("-" * 50)


if __name__ == "__main__":
    main()