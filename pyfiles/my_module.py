def bmi_index(height, weight):
    """计算BMI指数并返回数值和健康状况"""
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        status = "偏瘦"
    elif bmi < 24:
        status = "正常"
    elif bmi < 28:
        status = "超重"
    else:
        status = "肥胖"
    return bmi, status