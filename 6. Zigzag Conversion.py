class Solution:
    def convert(self, s: str, numRows: int) -> str :
        target_list = []
        for a in range(1, numRows+1) :
            target_list.append([])
        target_string = ""
        rowNum = 0
        i = 0
        while i < len(s) :
            if numRows <= 1 :
                target_list[rowNum].append(s[i])
                i += 1
                continue
            elif numRows > 1 :
                if rowNum != 0 and rowNum == numRows :
                    rowNum -= 1
                    for z in range(1, numRows-1) :
                        rowNum -= 1
                        target_list[rowNum].append(s[i])
                        i += 1
                        if i >= len(s) :
                            break
                    rowNum -= 1
                    continue
                else :
                    target_list[rowNum].append(s[i])
                    rowNum += 1
                    i += 1
        for j in range(1, numRows + 1) :
            for y in target_list[j-1] :
                target_string += y
        return target_string





                




        






