import os

def calculate_accuracy(output_file, answer_file):
    total_chars = 0
    correct_chars = 0
    total_sentences = 0
    correct_sentences = 0
    with open(output_file, 'r', encoding='utf-8') as f_out, \
         open(answer_file, 'r', encoding='utf-8') as f_ans:
        out_lines = f_out.readlines()
        ans_lines = f_ans.readlines()
        for line_out, line_ans in zip(out_lines, ans_lines):
            line_out = line_out.strip()
            line_ans = line_ans.strip()
            total_sentences += 1
            total_chars += len(line_ans)
            if line_out == line_ans:
                correct_sentences += 1
            for c_out, c_ans in zip(line_out, line_ans):
                if c_out == c_ans:
                    correct_chars += 1
    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0.0
    sentence_acc = (correct_sentences / total_sentences) * 100 if total_sentences > 0 else 0.0
    print("="*30)
    print(" 准确率统计报告")
    print("="*30)
    print(f"总字数 (Answer): {total_chars} 字")
    print(f"匹配字数:        {correct_chars} 字")
    print(f"--> 字准确率:    {char_acc:.2f}%\n")
    print(f"总句数 (Answer): {total_sentences} 句")
    print(f"完全匹配句数:    {correct_sentences} 句")
    print(f"--> 句准确率:    {sentence_acc:.2f}%")
    print("="*30)

if __name__ == "__main__":
    OUTPUT_PATH = './data/output3.txt'
    ANSWER_PATH = './data/answer.txt'
    calculate_accuracy(OUTPUT_PATH, ANSWER_PATH)