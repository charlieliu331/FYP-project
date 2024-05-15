def calculate_wer_and_diff(reference, hypothesis):
    # Preprocess sentences
    reference = ''.join(char.lower() for char in reference if char.isalnum() or char.isspace()).split()
    hypothesis = ''.join(char.lower() for char in hypothesis if char.isalnum() or char.isspace()).split()
    hypothesis = hypothesis[:min(len(hypothesis)-1,len(reference))+1]

    # Create a matrix to store the Levenshtein distances and operations
    #TODO change length of hypothesis
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    dp_matrix = [[(0, '') for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dp_matrix[i][0] = (i, 'D')  # Deletion
    for j in range(1, cols):
        dp_matrix[0][j] = (j, 'I')  # Insertion

    for i in range(1, rows):
        for j in range(1, cols):
            deletion = dp_matrix[i-1][j][0] + 1
            insertion = dp_matrix[i][j-1][0] + 1
            substitution = dp_matrix[i-1][j-1][0] + (reference[i-1] != hypothesis[j-1])
            if reference[i-1] == hypothesis[j-1]:
                dp_matrix[i][j] = (substitution, 'M')  # Match
            else:
                if substitution <= insertion and substitution <= deletion:
                    dp_matrix[i][j] = (substitution, 'S')  # Substitution
                elif insertion < deletion:
                    dp_matrix[i][j] = (insertion, 'I')  # Insertion
                else:
                    dp_matrix[i][j] = (deletion, 'D')  # Deletion

    # Reconstruct the path of operations
    i, j = len(reference), len(hypothesis)
    operations = []
    while i > 0 or j > 0:
        operation = dp_matrix[i][j][1]
        if operation == 'M' or operation == 'S':
            operations.append((operation, reference[i-1], hypothesis[j-1]))
            i -= 1
            j -= 1
        elif operation == 'D':
            operations.append((operation, reference[i-1], ''))
            i -= 1
        elif operation == 'I':
            operations.append((operation, '', hypothesis[j-1]))
            j -= 1

    operations.reverse()

    # Calculate WER
    wer = dp_matrix[len(reference)][len(hypothesis)][0] / len(reference)
    return wer, dp_matrix[len(reference)][len(hypothesis)][0], len(reference), operations





def extract(sentence):
    start_index = sentence.find("[/INST]") + len("[/INST]")
    extracted_part = sentence[start_index:].strip()
    return extracted_part

if __name__=="__main__":
    # Read sentences from files
    ref='groundtruth file'
    hyp='LLM output file'
    out='wer output file'
    with open(ref, 'r') as reference_file, open(hyp, 'r') as llama_file:
        reference_sentences = reference_file.readlines()
        hypothesis_sentences = llama_file.readlines()
    with open(out, 'w') as w:
        # Calculate total WER
        total_wer = 0
        total_errors = 0
        total_words = 0
        total_ser = 0

        for reference_sentence, hypothesis_sentence in zip(reference_sentences, hypothesis_sentences):
            wer, errors, words, diffs = calculate_wer_and_diff((reference_sentence), (hypothesis_sentence))
            total_wer += wer
            total_errors += errors
            total_words += words
                
            if errors != 0:
                total_ser += 1
                diff_output = []
                for op, ref_word, hyp_word in diffs:
                    if op == 'S':
                        diff_output.append(f"Substitution: {ref_word} -> {hyp_word}")
                    elif op == 'D':
                        diff_output.append(f"Deletion: {ref_word}")
                    elif op == 'I':
                        diff_output.append(f"Insertion: {hyp_word}")
                diff_str = '\n'.join(diff_output)
                w.write("GT: "+extract(reference_sentence)+'\n'+"Llama: "+extract(hypothesis_sentence)+'\n'+"Differences:\n"+diff_str+'\n\n')


    # Calculate average WER
    average_wer1 = total_wer / len(reference_sentences)
    average_wer2 = total_errors / total_words
    average_ser = total_ser / len(reference_sentences)

    print(f"Total Word Error Rate: {total_wer:.2f}")
    print(f"Average Word Error Rate: {average_wer1:.2f}")
    print(f"Average Word Error Rate: {average_wer2:.2f}")
    print(f"Sentence Error Rate: {average_ser:.2f}")
    print(f"Total Word Error: {total_errors}")
    print(f"Total Words: {total_words}")
    print(f"Number of Sentence Error: {total_ser}")



