import re

import re

class Extract_Reinsert:
    def __init__(self, input_file_path, llm_output_file_path, final_output_file_path):
        self.input_file_path = input_file_path
        self.llm_output_file_path = llm_output_file_path
        self.final_output_file_path = final_output_file_path

    def extract_words_with_punctuation(self, text):
        """Extract words followed by punctuation marks from the text, case-insensitively."""
        #pattern = r'\b(\w+)([\.,\?!;:\']+)' 
        pattern = r'\b(\w+)([\.,\?!;:]+)' #TODO temporarily exclude '
        return [(match.group(1) + match.group(2), match.start()) for match in re.finditer(pattern, text, re.IGNORECASE)]

    def find_closest_index(self, word, original_text, start_index=0):
        """Find the index of the word in the original text that is closest to the given start index, case-insensitively."""
        # clean_word = re.escape(word.rstrip('.,!?;:\''))
        clean_word = re.escape(word.rstrip('.,!?;:')) #TODO temporarily exclude '
        indices = [m.start() for m in re.finditer(r'\b' + clean_word + r'\b', original_text, re.IGNORECASE)]
        if not indices:
            return -1
        closest_index = min(indices, key=lambda x: abs(x - start_index))
        return closest_index

    def insert_punctuation(self, original_text, words_with_punctuation):
        """Insert punctuation marks back into the original text at the position of their closest corresponding words."""
        new_text_list = list(original_text)
        offset = 0

        for word, original_position in words_with_punctuation:
            closest_index = find_closest_index(word, original_text, original_position)
            if closest_index >= 0:
                # insert_position = closest_index + len(word.rstrip('.,\'!?;:')) + offset
                insert_position = closest_index + len(word.rstrip('.,!?;:')) + offset #TODO temporarily '
                new_text_list.insert(insert_position, word[-1])
                offset += 1

        return ''.join(new_text_list)

    def generate(self):
        with open(self.input_file_path, 'r') as input_file, open(self.llm_output_file_path, 'r') as llm_output_file, open(self.final_output_file_path, 'w') as final_output_file:
            for input_line, llm_output_line in zip(input_file, llm_output_file):
                words_with_punctuation = self.extract_words_with_punctuation(llm_output_line)
                modified_input = self.insert_punctuation(input_line.strip(), words_with_punctuation)
                final_output_file.write(modified_input + '\n')


input_file = 'the path of input txt file to be fed into LLM for punctuation restoration'
llm_output_file = 'the path of llm output text file'
final_output_file = 'the path of file you want to output'

restorer = PunctuationRestorer(input_file, llm_output_file, final_output_file)
restorer.generate()
