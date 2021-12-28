# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        l1_value = 0
        l2_value = 0
        
        tmp = 1
        
        while l1 :
            l1_value += l1.val * tmp
            l1 = l1.next
            tmp  = tmp * 10
        
        tmp = 1
        while l2 :
            l2_value += l2.val * tmp
            l2 = l2.next
            tmp = tmp * 10
        
        target_value = l1_value + l2_value
        
        header = None
        linked_list = None
        
        for c in str(target_value):
            if not header :
                header = ListNode(int(c))
                linked_list = header
            else :
                curNode = ListNode(int(c), linked_list)
                linked_list = curNode
                
        return linked_list
        