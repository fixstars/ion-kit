from ionpy import Param


def test_param():

    p1 = Param(key='iamkey1', val="IAMKEY")  # 'IAMKEY'
    p2 = Param(key='iamkey2', val="iamkey")  # 'iamkey'
    p3 = Param(key='iamkey3', val=1)  # '1'
    p4 = Param(key='iamkey4', val=0.1)  # '0.1'
    p5 = Param(key='iamkey5', val=True)   # 'true'
    p6 = Param(key='iamkey6', val=False)  # 'false'
