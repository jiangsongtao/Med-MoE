import argparse
import json
import collections
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to ground truth file')
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file')
    parser.add_argument('--output', type=str, default="wrong_answers.json", help='path to output file for wrong answers')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred):    
    scores = collections.defaultdict(list)
    closed_scores = collections.defaultdict(list)
    closed_questions_count=0
    closed_questions_correct=0
    wrong_answers = []

    for gt_item, pred_item in zip(gt, pred):
        gt_results = gt_item.get('conversations', gt_item.get('conversatons'))
        gt_value = gt_results[1]['value'].lower()
        pred_value = pred_item['text'].lower()
        answer_type = gt_item['answer_type']
        if answer_type == 'open' or answer_type == 'OPEN':
            scores['exact_match'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            scores['f1'].append(f1_score)
            scores['precision'].append(precision)
            scores['recall'].append(recall)

            weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]  
            bleu_scores = []
            for w in weights:
                bleu_score = sentence_bleu([gt_value.split()], pred_value.split(), weights=w, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu_score)
            scores['bleu_scores'].append(bleu_scores)
            
        elif answer_type == 'close' or answer_type == 'CLOSED' :
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            closed_scores['f1'].append(f1_score)
            closed_scores['precision'].append(precision)
            closed_scores['recall'].append(recall)
            closed_questions_count += 1
            if gt_value not in pred_value:  # Check if gt_value is not contained within pred_value
                wrong_answers.append({'prompt': pred_item['prompt'], 'correct_answer': gt_value, 'predicted_answer': pred_value})
            else:
                closed_questions_correct += 1  # Update the count of correct answers

    exact_match_avg = sum(scores['exact_match']) / len(scores['exact_match'])
    f1_score_avg = sum(scores['f1']) / len(scores['f1'])
    precision_avg = sum(scores['precision']) / len(scores['precision'])
    recall_avg = sum(scores['recall']) / len(scores['recall'])
    bleu_scores_avg = [sum(score_list) / len(score_list) for score_list in zip(*scores['bleu_scores'])]

    closed_score = (closed_questions_correct / closed_questions_count * 100) if closed_questions_count else 0
    closed_f1_score_avg = sum(closed_scores['f1']) / len(closed_scores['f1'])
    closed_precision_avg = sum(closed_scores['precision']) / len(closed_scores['precision'])
    closed_recall_avg = sum(closed_scores['recall']) / len(closed_scores['recall'])

    results_table = tabulate(
        [
            ['Exact Match Score', exact_match_avg*100],
            ['F1 Score', f1_score_avg*100],
            ['Precision', precision_avg*100],
            ['Recall', recall_avg*100],
            ['BLEU Score (Weight 1)', bleu_scores_avg[0]*100],
            ['BLEU Score (Weight 2)', bleu_scores_avg[1]*100],
            ['BLEU Score (Weight 3)', bleu_scores_avg[2]*100],
            ['yes/no accuracy', closed_score], 
            ['Closed F1 Score', closed_f1_score_avg*100],
            ['Closed Precision', closed_precision_avg*100],
            ['Closed Recall', closed_recall_avg*100],
        ],
        headers=['Metric', 'Performance (%)']
    )
    
    with open(args.output, 'w') as f:
        json.dump(wrong_answers, f, indent=4)

    return results_table

if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    results = evaluate(gt, pred)
    print(results)
