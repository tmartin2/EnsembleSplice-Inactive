import pandas as pd
from random import seed, shuffle



def main():
    #seed(1232423423)
    # HOMO
    # targets = [
    #     '../../Homo/Original/negative_DNA_seqs_donor_hs.fa',
    #     '../../Homo/Original/positive_DNA_seqs_donor_hs.fa',
    #     '../../Homo/Original/negative_DNA_seqs_acceptor_hs.fa',
    #     '../../Homo/Original/positive_DNA_seqs_acceptor_hs.fa',
    # ]
    # train_dests = [
    #     '../../Homo/Train/neg_donor_hs_train.fa',
    #     '../../Homo/Train/pos_donor_hs_train.fa',
    #     '../../Homo/Train/neg_acceptor_hs_train.fa',
    #     '../../Homo/Train/pos_acceptor_hs_train.fa',
    # ]
    # test_dests = [
    #     '../../Homo/Test/neg_donor_hs_test.fa',
    #     '../../Homo/Test/pos_donor_hs_test.fa',
    #     '../../Homo/Test/neg_acceptor_hs_test.fa',
    #     '../../Homo/Test/pos_acceptor_hs_test.fa',
    # ]
    #
    # for index, target in enumerate(targets):
    #     with open(target, 'r') as f:
    #         lines = f.readlines()
    #         shuffle(lines)
    #         f.close()
    #     train = lines[:round(len(lines)*0.8)]
    #     test = lines[round(len(lines)*0.8):]
    #     with open(train_dests[index], 'w') as f:
    #         f.writelines(train)
    #         f.close()
    #     with open(test_dests[index], 'w') as f:
    #         f.writelines(test)
    #         f.close()

    # HS3D

  targets = [
      './HS3D/Original/Acceptor_Negative01.txt',
      './HS3D/Original/Donor_Negative01.txt',
      './HS3D/Original/Acceptor_Positive.txt',
      './HS3D/Original/Donor_Positive.txt',
  ]
  train_dests = [
      './HS3D/SubsetTrain/Acceptor_Train_Negative01.txt',
      './HS3D/SubsetTrain/Donor_Train_Negative01.txt',
      './HS3D/SubsetTrain/Acceptor_Train_Positive.txt',
      './HS3D/SubsetTrain/Donor_Train_Positive.txt',
  ]
  test_dests = [
      './HS3D/SubsetTest/Acceptor_Test_Negative01.txt',
      './HS3D/SubsetTest/Donor_Test_Negative01.txt',
      './HS3D/SubsetTest/Acceptor_Test_Positive.txt',
     './HS3D/SubsetTest/Donor_Test_Positive.txt',
  ]
  # read in first acceptor pos, neg and first donor pos, neg
  # take 10000 instances random sampled
  # for index, target in enumerate(targets):
  with open('./HS3D/SubsetTest/Acceptor_Test_Positive.txt', 'r') as f:
      lines = f.readlines()[4:]
      lines = lines[:-1] # remove the last incomplete line
      lines = [elt.split(':')[1].replace('\n','').replace(' ','') for elt in lines]
      seed(123432)
      shuffle(lines)
      f.close()
  # if 'Neg' in target:
  #     train = lines[:10000]
  #     test = lines[10000:12000]
  # else:
      # train = lines[:round(len(lines)*0.8)]
      # test = lines[round(len(lines)*0.8):]
  with open('issCNN_acceptor_positive_test.fa', 'w') as f:
      for index, x in enumerate(lines):
          f.write(f'>seq{index+1}\n{x}\n')
      f.close()
      #assert len([elt for elt in train if '(' not in elt])==0,f'fuck'
      # with open(train_dests[index], 'w') as f:
      #     f.writelines(train)
      #     f.close()
      # with open(test_dests[index], 'w') as f:
      #     f.writelines(test)
      #     f.close()





    # # CE
    # targets = [
    #     './CE/Original/Acceptor_All.txt',
    #     './CE/Original/Donor_All.txt',
    # ]
    # train_dests = [
    #     './CE/Train/Acceptor_Train.txt',
    #     './CE/Train/Donor_Train.txt',
    # ]
    # test_dests = [
    #     './CE/Test/Acceptor_Test.txt',
    #     './CE/Test/Donor_Test.txt',
    # ]
    #
    # for index, target in enumerate(targets):
    #     with open(target, 'r') as f:
    #         lines = f.readlines()
    #         shuffle(lines)
    #         f.close()
    #     train = lines[:round(len(lines)*0.8)]
    #     test = lines[round(len(lines)*0.8):]
    #     with open(train_dests[index], 'w') as f:
    #         f.writelines(train)
    #         f.close()
    #     with open(test_dests[index], 'w') as f:
    #         f.writelines(test)
    #         f.close()

    # # HOMO
    # targets = [
    #     '../../Oryza/Original/negative_DNA_seqs_donor_oriza.fa',
    #     '../../Oryza/Original/positive_DNA_seqs_donor_oriza.fa',
    #     '../../Oryza/Original/negative_DNA_seqs_acceptor_oriza.fa',
    #     '../../Oryza/Original/positive_DNA_seqs_acceptor_oriza.fa',
    # ]
    # train_dests = [
    #     '../../Oryza/Train/neg_donor_oriza_train.fa',
    #     '../../Oryza/Train/pos_donor_oriza_train.fa',
    #     '../../Oryza/Train/neg_acceptor_oriza_train.fa',
    #     '../../Oryza/Train/pos_acceptor_oriza_train.fa',
    # ]
    # test_dests = [
    #     '../../Oryza/Test/neg_donor_oriza_test.fa',
    #     '../../Oryza/Test/pos_donor_oriza_test.fa',
    #     '../../Oryza/Test/neg_acceptor_oriza_test.fa',
    #     '../../Oryza/Test/pos_acceptor_oriza_test.fa',
    # ]
    #
    # for index, target in enumerate(targets):
    #     with open(target, 'r') as f:
    #         lines = f.readlines()
    #         shuffle(lines)
    #         f.close()
    #     train = lines[:round(len(lines)*0.8)]
    #     test = lines[round(len(lines)*0.8):]
    #     with open(train_dests[index], 'w') as f:
    #         f.writelines(train)
    #         f.close()
    #     with open(test_dests[index], 'w') as f:
    #         f.writelines(test)
    #         f.close()




if __name__ == '__main__':
    main()
