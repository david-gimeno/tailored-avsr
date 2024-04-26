import subprocess

def compute_bootstrap_wer(path):
    wer_command = ['./src/evaluation/tasas/tasas', '-f', "#", '-s', " ", "-ie", path]
    cer_command = ['./src/evaluation/tasas/tasas', '-f', "#", "-ie", path]

    wer = float(subprocess.check_output(wer_command).strip())
    cer = float(subprocess.check_output(cer_command).strip())

    ci_wer_command = ['./src/evaluation/tasas/tasasIntervalo', '-f', "#", '-s', " ", "-ie", path]
    ci_cer_command = ['./src/evaluation/tasas/tasasIntervalo', '-f', "#", "-ie", path]

    ci_wer = float(subprocess.check_output(ci_wer_command).decode('utf-8').split('+-')[1].strip())
    ci_cer = float(subprocess.check_output(ci_cer_command).decode('utf-8').split('+-')[1].strip())

    return wer, cer, ci_wer, ci_cer
