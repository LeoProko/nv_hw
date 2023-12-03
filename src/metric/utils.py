import editdistance


def calc_wer(target_text: str, pred_text: str):
    if target_text == "":
        if pred_text == "":
            return 0
        return 1

    return editdistance.eval(target_text.split(), pred_text.split()) / len(
        target_text.split()
    )


def calc_cer(target_text: str, pred_text: str):
    if target_text == "":
        if pred_text == "":
            return 0
        return 1

    return editdistance.eval(target_text, pred_text) / len(target_text)
