import re


def get_think(output: str) -> str:
    # Use re.findall to capture every <think>...</think> block
    matches = re.findall(r'<think>(.*?)</think>', output, re.DOTALL)

    # If there are matches, keep the last one; otherwise fall back to full output
    if matches:
        think_part = matches[-1].strip()
    else:
        think_part = output.strip()
    
    return think_part


def get_option_letter(content_answer: str) -> str:
    # Pull a single letter A-Z from the content via regex
    student_answer_match = re.search(r'[A-Z]', content_answer.upper())
    student_answer = student_answer_match.group(0) if student_answer_match else ""
    return student_answer.upper().strip()


def get_answer(output: str) -> str:
    # Use re.findall to capture every <answer>...</answer> block
    matches = re.findall(r'<answer>(.*?)</answer>', output, re.DOTALL)

    # If there are matches, keep the last one; otherwise use the final sentence
    if matches:
        content_answer = matches[-1].strip()
    else:
        sentences = re.split(r'[。！？.!?]', output)
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        content_answer = sentences[-1] if sentences else ""

    return content_answer
