import json
import re
import argparse

def read_json(pth):
    with open(pth, 'r') as f:
        return json.load(f)
def save_json(data, pth):
    with open(pth, 'w') as f:
        json.dump(data, f, indent=4)

def acc_gsm8k(preds,golds):
    print("preds[0]:{}".format(preds[0]))
    print("golds[0]:{}".format(golds[0]))
    #assert 1==0
    pattern = r"###\s*(-?\d+)"
    ans_pattern = r'answer is (\d+)'
    all_pattern = r'(-?\d+(\.\d+)?)\D*$'
    g = []
    # for gold in golds:
    #     match = re.search(pattern, gold)
    #     if match:
    #         g.append(int(match.group(1)))
    #     else:
    #         print("!!!!!!!!{}: not match".format(gold))
    print("golds:{}".format(golds))
    #
    g = [int(re.search(pattern, gold).group(1)) for gold in golds]
    # print(len(g))
    reslog = []
    correct = 0
    sum = 0
    print(len(g))
    for i in range(len(g)):
        sum += 1
        pred=preds[i]
        if isinstance(pred, list):
            pred = pred[0]
        print("pred:{}".format(pred))
        print("g[i]:{}".format(g[i]))

        pred=pred.split('\n\n')[0]
        
        match = re.search(ans_pattern, pred.replace(",", ""))
        if match:
            tmp_num = int(float(match.group(1)))
            if tmp_num == g[i]:
                correct += 1
        else:
            all_m = re.search(all_pattern, pred.replace(",", ""))
            if all_m:
                try:
                   tmp_num = int(float(all_m.group(1)))
                except :
                   print("error")
                if tmp_num == g[i]:
                    correct += 1
            else:
                tmp_num = 0
        reslog.append("id:{}\tpred:{}\tgold:{}".format(i,tmp_num, g[i]))
    acc = float(correct) / sum
    return acc, reslog

def acc_svamp(preds,golds):
    ans_pattern = r'answer is (\d+)'
    all_pattern = r'(-?\d+(\.\d+)?)\D*$'
    sum, correct = 0, 0
    reslog = []
    for i,gold in enumerate(golds):
        pred = preds[i]
        match = re.search(ans_pattern, pred)
        sum += 1
        if match:
            tmp_num = float(match.group(1))
            if abs(tmp_num - gold)<1e-6:
                correct += 1
        else:
            all_m = re.search(all_pattern, pred)
            if all_m:
                tmp_num = float(all_m.group(1))
                if abs(tmp_num - gold)<1e-6:
                    correct += 1
            else:
                tmp_num = 0
        reslog.append("id:{}\tpred:{}\tgold:{}".format(i,tmp_num, gold))
        # if i==0:
        #     print(pred)
        #     print(reslog)
    acc = float(correct) / sum
    return acc, reslog

def acc_aqua(preds,golds):
    ans_pattern1 = r'answer is ([A-Za-z])'
    ans_pattern2 = r'answer: ([A-Za-z])'
    all_pattern = r'[A-Z]'
    sum, correct = 0, 0
    reslog = []
    # print(preds[0])
    # print(golds[0])
    for pred,gold in zip(preds,golds):
        
        if pred==gold:
            correct += 1
        reslog.append(f"id:{sum}\tpred:{pred}\tgold:{gold}")
        # if sum==0:
            # print(reslog)
        sum += 1
    # for i, pred in enumerate(preds):
    #     gold = golds[i]
    #     match1 = re.search(ans_pattern1, pred, re.IGNORECASE)
    #     match2 = re.search(ans_pattern2, pred, re.IGNORECASE)
    #     sum += 1
    #     if match1:
    #         tmp_choice = match1.group(1)
    #         if tmp_choice == gold:
    #             correct += 1
    #     elif match2:
    #         tmp_choice = match2.group(1)
    #         if tmp_choice == gold:
    #             correct += 1
    #     else:
    #         all_m = re.findall(all_pattern, pred)
    #         if all_m:
    #             tmp_choice = all_m[-1]
    #             if tmp_choice == gold:
    #                 correct += 1
    #         else:
    #             tmp_choice = 'Z'
        
    #     reslog.append("id:{}\tpred:{}\tgold:{}".format(i,tmp_choice, gold))
    #     if i==10:
    #         print(reslog)
    acc = float(correct) / sum
    return acc, reslog

def acc_asdiv(preds,golds):
    ans_pattern = r'answer is (\d+)'
    all_pattern = r'(-?\d+(\.\d+)?)\D*$'
    sum, correct = 0, 0
    reslog = []
    for i,gold in enumerate(golds):
        pred = preds[i]
        match = re.search(ans_pattern, pred)
        sum += 1
        if match:
            tmp_num = match.group(1)
            if tmp_num in gold:
                correct += 1
        else:
            all_m = re.search(all_pattern, pred)
            if all_m:
                tmp_num = all_m.group(1)
                if tmp_num in gold:
                    correct += 1
            else:
                tmp_num = 0
        reslog.append("id:{}\tpred:{}\tgold:{}".format(i,tmp_num, gold))
    acc = float(correct) / sum
    return acc, reslog

def acc_folio(preds,golds):
    def get_last_word(text):
        pattern = r'(true|false|uncertain)'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            return last_match.group(0)
        else:
            return None
        
    reslog = []
    correct, sum = 0, 0
    for i,pred in enumerate(preds):
        gold=golds[i].lower()
        # pred=item["answer"]
        # pred=pred.split('\n\n')[0]
        pred=pred.split('<|eot_id|>')[0]
        pred=pred.split('Question: The following is a first-order logic (FOL) problem.')[0].strip('\n')
        pred=get_last_word(pred)
        if pred and pred.lower()==gold:
            correct+=1
        sum+=1
        reslog.append("id:{}\tpred:{}\tgold:{}".format(i,pred, gold))
    acc=float(correct)/sum
    return acc, reslog

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--response_file", type=str)
    # parser.add_argument("--gold_file", type=str, default="output/vanilla/gsm8k/gsm8k_ans.json")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()

    gold_file = {'gsm8k': 'datasets/gsm8k/test.json',
                'aqua': 'index_data/aqua/aqua/test.json',
                'svamp': 'index_data/svamp/svamp/test.json',
                'asdiv': 'index_data/asdiv/asdiv/test.json',
                'folio': 'index_data/folio/folio/test.json',
                }
    responses = read_json(args.response_file)
    golds = read_json(gold_file[args.task])
    assert len(responses) == len(golds), (len(responses), len(golds))
    golds = responses
    if responses[0].get("generated"):
        preds = [response["generated"] for response in responses]
        if isinstance(preds[0],list):
            preds = [pred[0] for pred in preds]
        preds = [pred.split('\n\n')[0] for pred in preds]
    elif responses[0].get("pred"):
        preds = [response["pred"] for response in responses]
        if isinstance(preds[0],list):
            preds = [pred[0] for pred in preds]
        preds = [pred.split('\n\n')[0] for pred in preds]
    elif responses[0].get("answer"):
        preds = [response["answer"] for response in responses]
        if isinstance(preds[0],list):
            preds = [pred[0] for pred in preds]
        preds = [pred.split('\n\n')[0] for pred in preds]

    
    # print(preds[0])
    if args.task=='gsm8k':
        golds = [gold["answer"] for gold in golds]
        acc, reslog = acc_gsm8k(preds, golds)
    if args.task=='aqua':
        golds = [gold["correct"] for gold in golds]
        acc, reslog = acc_aqua(preds, golds)
    if args.task=='svamp':
        golds = [gold["Answer"] for gold in golds]
        acc, reslog = acc_svamp(preds, golds)
    if args.task=='asdiv':
        golds = [gold["Answer"] for gold in golds]
        acc, reslog = acc_asdiv(preds, golds)
    if args.task=='folio':
        golds = [gold["label"] for gold in golds]
        acc, reslog = acc_folio(preds, golds)
    print("=========save to:{}".format(args.output_file))
    print("=========acc result:{}".format(acc))
    if args.output_file!="":
        save_json(reslog, args.output_file)
    
if __name__ == "__main__":
    main()