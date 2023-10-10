import torch

from typing import List


def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]: # 计算准确率的前k个预测的元组，target是真实的标签
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # 使用topk方法找到前k个最大的值，返回两个tensor，第一个是值，第二个是索引
        pred = pred.t() # 每列包含一个样本的前maxk个最可能的预测，需要验证一下
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # 将预测结果pred与真实标签target进行比较，生成一个布尔张量correct，其中元素为True表示预测正确，False表示预测错误

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


'''
下面是我写的一个测试函数
'''
def main():
    from torch.nn.functional import softmax
    output = torch.randn(5, 10) # 5个样本，10个类别
    output = softmax(output, dim=-1)
    label = torch.tensor([1, 1, 3, 5, 7])
    print(accuracy(output, label, topk=(1, 2, 3,4,10)))
    

    
    
#if __name__ == '__main__':
    #main()









