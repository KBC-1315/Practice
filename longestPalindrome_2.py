class Solution:
    def longestPalindrome(self, s: str) -> str:
        def expand(left: int, right: int) -> str:
            while left >= 0 and right <= len(s) and s[left] == s[right - 1]:
                left -= 1
                right += 1
            return s[left + 1 : right - 1] #위에서 한번 더 빼/더해 주었으므로 되돌려놔야 된다.

        if len(s) < 2 and s == s[::-1]: # 해당 사항이 없을 때는 빠르게 리턴해버린다.
            return s
            
        result = "" # result에는 계속해서 가장 긴 팰린드롬이 갱신되어 저장될 것임

        for i in range(len(s) - 1):
            # expaned(i, i+1)는 홀수 팰린드롬 판단, expaned(i, i+2)는 짝수 팰린드롬 판단
            result = max(result, expand(i, i + 1), expand(i, i + 2), key=len)
        return result