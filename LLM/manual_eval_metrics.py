'''change the punctuation classes if necessary'''
import string


def align_tokens(predicted_, reference_):
    predicted = [token.lower() for token in predicted_]
    reference = [token.lower() for token in reference_]
    
    predicted = predicted[:len(reference)+1]

    # Initialize a matrix for dynamic programming
    dp_matrix = [[0] * (len(reference) + 1) for _ in range(len(predicted) + 1)]

    # Fill in the matrix based on the Needleman-Wunsch algorithm
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(reference) + 1):
            if predicted[i - 1] == reference[j - 1]:
                dp_matrix[i][j] = dp_matrix[i - 1][j - 1] + 1
            else:
                dp_matrix[i][j] = max(dp_matrix[i - 1][j], dp_matrix[i][j - 1])

    # Backtrack to find the aligned sequences
    aligned_predicted = []
    aligned_reference = []
    i, j = len(predicted), len(reference)
    while i > 0 and j > 0:
        if predicted[i - 1] == reference[j - 1]:
            aligned_predicted.insert(0, predicted[i - 1])
            aligned_reference.insert(0, reference[j - 1])
            i -= 1
            j -= 1
        elif dp_matrix[i - 1][j] > dp_matrix[i][j - 1]:
            aligned_predicted.insert(0, predicted[i - 1])
            aligned_reference.insert(0, '|SPACE|')  # Insert a space for unmatched token in reference
            i -= 1
        else:
            aligned_predicted.insert(0, '|SPACE|')  # Insert a space for unmatched token in predicted
            aligned_reference.insert(0, reference[j - 1])
            j -= 1

    # Handle any remaining tokens
    while i > 0:
        aligned_predicted.insert(0, predicted[i - 1])
        aligned_reference.insert(0, '|SPACE|')
        i -= 1
    while j > 0:
        aligned_predicted.insert(0, '|SPACE|')
        aligned_reference.insert(0, reference[j - 1])
        j -= 1

    return aligned_predicted, aligned_reference

def calculate_metrics(groundtruth_file, eval_file):
    # Read the content of the groundtruth and evaluation files
    with open(groundtruth_file, 'r', encoding='utf-8') as file:
        groundtruth_text = file.readlines()
    with open(eval_file, 'r', encoding='utf-8') as file:
        eval_text = file.readlines()

    # Define a function to calculate precision, recall, and F1 score
    def calculate_prf(true_positives, false_positives, false_negatives):
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1_score

    # Initialize variables to count true positives, false positives, and false negatives
    tp_comma, fp_comma, fn_comma = 0, 0, 0
    tp_period, fp_period, fn_period = 0, 0, 0

    #TODO addin
    tp_ex, fp_ex, fn_ex = 0, 0, 0
    tp_q, fp_q, fn_q = 0, 0, 0
    tp_all_punct, fp_all_punct, fn_all_punct = 0, 0, 0

    assert(len(groundtruth_text)==len(eval_text))
    print(len(groundtruth_text))
    # Tokenize the texts into words
    # groundtruth_tokens = [i.split(' ') for i in groundtruth_text]
    # eval_tokens = [i.split(' ') for i in eval_text]
    groundtruth_tokens = []
    eval_tokens = []
    aligned_sequence = [align_tokens(ev.split(),gt.split()) for ev,gt in zip(eval_text,groundtruth_text)]
    with open("./aligned_0213.txt",'w') as f1:
        for aligned_predicted, aligned_reference in aligned_sequence:
            eval_tokens.append(aligned_predicted)
            f1.write("LL: "+' '.join(aligned_predicted)+" \n")
            groundtruth_tokens.append(aligned_reference)
            f1.write("GT: "+' '.join(aligned_reference)+" \n")

    print(len(groundtruth_tokens),len(eval_tokens))

    token_number_mismatch=[]
    fp_comma_txt=[]
    fn_comma_txt=[]
    fp_period_txt=[]
    fn_period_txt=[]
    #TODO addin
    fp_ex_txt=[]
    fn_ex_txt=[]
    fp_q_txt=[]
    fn_q_txt=[]

    for i in range(len(groundtruth_tokens)):
        if len(groundtruth_tokens[i])==len(eval_tokens[i]):
            for gt_token, eval_token in zip(groundtruth_tokens[i], eval_tokens[i]):
                
                if ',' in gt_token and ',' in eval_token:
                    tp_comma += 1
                    tp_all_punct += 1
                elif ',' in eval_token:
                    fp_comma += 1
                    fp_all_punct += 1
                    if i not in fp_comma_txt:
                        fp_comma_txt.append(i)
                elif ',' in gt_token:
                    fn_comma += 1
                    fn_all_punct += 1
                    if i not in fn_comma_txt:
                        fn_comma_txt.append(i)

                if '.' in gt_token and '.' in eval_token:
                    tp_period += 1
                    tp_all_punct += 1
                elif '.' in eval_token:
                    fp_period += 1
                    fp_all_punct += 1
                    if i not in fp_period_txt:
                        fp_period_txt.append(i)
                elif '.' in gt_token:
                    fn_period += 1
                    fn_all_punct += 1
                    if i not in fn_period_txt:
                        fn_period_txt.append(i)
                    
                # if '!' in gt_token and '!' in eval_token:
                #     tp_period += 1
                #     tp_all_punct += 1
                # elif '!' in eval_token:
                #     fp_period += 1
                #     fp_all_punct += 1
                #     if i not in fp_period_txt:
                #         fp_period_txt.append(i)
                # elif '!' in gt_token:
                #     fn_period += 1
                #     fn_all_punct += 1
                #     if i not in fn_period_txt:
                #         fn_period_txt.append(i)
                # if '?' in gt_token and '?' in eval_token:
                #     tp_period += 1
                #     tp_all_punct += 1
                # elif '?' in eval_token:
                #     fp_period += 1
                #     fp_all_punct += 1
                #     if i not in fp_period_txt:
                #         fp_period_txt.append(i)
                # elif '?' in gt_token:
                #     fn_period += 1
                #     fn_all_punct += 1
                #     if i not in fn_period_txt:
                #         fn_period_txt.append(i)

                #TODO addin
                if '!' in gt_token and '!' in eval_token:
                    tp_ex += 1
                    tp_all_punct += 1
                elif '!' in eval_token:
                    fp_ex += 1
                    fp_all_punct += 1
                    if i not in fp_ex_txt:
                        fp_ex_txt.append(i)
                elif '!' in gt_token:
                    fn_ex += 1
                    fn_all_punct += 1
                    if i not in fn_ex_txt:
                        fn_ex_txt.append(i)
                if '?' in gt_token and '?' in eval_token:
                    tp_q += 1
                    tp_all_punct += 1
                elif '?' in eval_token:
                    fp_q += 1
                    fp_all_punct += 1
                    if i not in fp_q_txt:
                        fp_q_txt.append(i)
                elif '?' in gt_token:
                    fn_q += 1
                    fn_all_punct += 1
                    if i not in fn_q_txt:
                        fn_q_txt.append(i)
        else: token_number_mismatch.append(i)

        # Calculate metrics for all punctuation
        # if set(gt_token) == set(eval_token):
        #     tp_all_punct += 1
        # else:
        #     fp_all_punct += len(set(eval_token) - set(gt_token))
        #     fn_all_punct += len(set(gt_token) - set(eval_token))
    with open('./error_output_llama2.txt','w') as f:
        # f.write("------------- TOKEN NUMBER MISMATCH -------------\n")
        # for i in token_number_mismatch:
        #     f.write("Denormalizer:\n")
        #     f.write(eval_text[i])
        #     f.write("LIBRIHEAVY:\n")
        #     f.write(groundtruth_text[i])
        f.write("\n\n------------- FP COMMA -------------\n")
        for i in fp_comma_txt:
            f.write("LL:\n")
            f.write(eval_text[i])
            f.write("GT:\n")
            f.write(groundtruth_text[i])
        f.write("\n\n------------- FN COMMA -------------\n")
        for i in fn_comma_txt:
            f.write("LL:\n")
            f.write(eval_text[i])
            f.write("GT:\n")
            f.write(groundtruth_text[i])
        f.write("\n\n------------- FP PERIOD -------------\n")
        for i in fp_period_txt:
            f.write("LL:\n")
            f.write(eval_text[i])
            f.write("GT:\n")
            f.write(groundtruth_text[i])
        f.write("\n\n------------- FN PERIOD -------------\n")
        for i in fn_period_txt:
            f.write("LL:\n")
            f.write(eval_text[i])
            f.write("GT:\n")
            f.write(groundtruth_text[i])


    # Calculate precision, recall, and F1 score for comma
    precision_comma, recall_comma, f1_comma = calculate_prf(tp_comma, fp_comma, fn_comma)

    # Calculate precision, recall, and F1 score for period
    precision_period, recall_period, f1_period = calculate_prf(tp_period, fp_period, fn_period)

    #TODO addin
    precision_ex, recall_ex, f1_ex = calculate_prf(tp_ex, fp_ex, fn_ex)
    precision_q, recall_q, f1_q = calculate_prf(tp_q, fp_q, fn_q)
    # Calculate precision, recall, and F1 score for all punctuation
    precision_all_punct, recall_all_punct, f1_all_punct = calculate_prf(tp_all_punct, fp_all_punct, fn_all_punct)

    # Calculate macro-F1 score
    # macro_f1 = (f1_comma + f1_period + f1_all_punct) / 3
    macro_f1 = (f1_comma + f1_period + f1_q+ f1_ex + f1_all_punct) / 5

    return {
        "Comma": {"Precision": precision_comma, "Recall": recall_comma, "F1 Score": f1_comma},
        "Period": {"Precision": precision_period, "Recall": recall_period, "F1 Score": f1_period},
        "Exclamation": {"Precision": precision_ex, "Recall": recall_ex, "F1 Score": f1_ex},
        "Question": {"Precision": precision_q, "Recall": recall_q, "F1 Score": f1_q},
        "All Punctuation": {"Precision": precision_all_punct, "Recall": recall_all_punct, "F1 Score": f1_all_punct},
        "Macro F1": macro_f1
    }


#TODO add path here
results = calculate_metrics('groudtruth text file','the extracted and reinserted file of LLM output')

for label, metrics in results.items():
    if label != "Macro F1":
        print(label)
        print(f"Precision: {metrics['Precision']:.2f}")
        print(f"Recall: {metrics['Recall']:.2f}")
        print(f"F1 Score: {metrics['F1 Score']:.2f}")
        print()

print(f"Macro F1 Score: {results['Macro F1']:.2f}")


