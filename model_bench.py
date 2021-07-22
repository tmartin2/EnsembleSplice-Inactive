import os


# Benchmark issCNN

targets_hs3d = [
    './issCNN/HS3D_SubsetTest_Results/issCNN_acceptor_negative_results.txt',
    './issCNN/HS3D_SubsetTest_Results/issCNN_acceptor_positive_results.txt',
    './issCNN/HS3D_SubsetTest_Results/issCNN_donor_negative_results.txt',
    './issCNN/HS3D_SubsetTest_Results/issCNN_donor_positive_results.txt'
]
don_correct = 0
don_total = 0
acc_correct = 0
acc_total = 0
for target in targets_hs3d:
    with open(target, 'r') as f:
        if 'acceptor' in target:
            if 'negative' in target:
                lines = f.readlines()
                good = [elt for elt in lines if 'Not' in elt]
                acc_correct+=len(good)
                acc_total+=len(lines)
            if 'positive' in target:
                lines = f.readlines()
                good = [elt for elt in lines if 'Not' not in elt]
                acc_correct+=len(good)
                acc_total+=len(lines)
        if 'donor' in target:
            if 'negative' in target:
                lines = f.readlines()
                good = [elt for elt in lines if 'Not' in elt]
                don_correct+=len(good)
                don_total+=len(lines)
            if 'positive' in target:
                lines = f.readlines()
                good = [elt for elt in lines if 'Not' not in elt]
                don_correct+=len(good)
                don_total+=len(lines)
print(f'issCNN HS3D SubsetTest Donor Accuracy = {don_correct/don_total}')
print(f'issCNN HS3D SubsetTest Acceptor Accuracy = {acc_correct/acc_total}')
