import os
import subprocess

wer_command = ['./tasasIntervalo', '-f', "#", '-s', " ", '../01_decode/LIP-RTVE/DEV_LIP-RTVE_007.txt']
wer = subprocess.check_output(wer_command)
print(wer.decode('utf-8').split('+-')[1].strip())
"""
cer_command = ['./tasas', '-f', "#", '../01_decode/LIP-RTVE/DEV_LIP-RTVE_007.txt']
cer = subprocess.check_output(cer_command).strip()

print(wer, cer)

import jiwer

f = open('../01_decode/LIP-RTVE/DEV_LIP-RTVE_007.txt', 'r').readlines()
wers = []
cers = []
for line in f:
    ref = line.strip().split('#')[0]
    hyp = line.strip().split('#')[1]
    wers.append(jiwer.wer(ref, hyp))
    cers.append(jiwer.cer(ref, hyp))

print(sum(wers)/len(wers), sum(cers)/len(cers))
"""
