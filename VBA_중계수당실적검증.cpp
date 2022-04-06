Sub 중계실적검증()
Dim cnt As Integer
Dim name As String
Dim name2 As String
Dim name3 As String
Dim target_amount As Integer
Dim row_num3 As Integer
Dim row_num2 As Integer

Sheets.Add before:=Sheets(1)
Sheets(1).name = "중계실적 검증"

Worksheets("중계실적").Activate
row_num3 = 6
row_num2 = 1
cnt = Application.WorksheetFunction.CountA(Columns("A")) - 5

Worksheets("중계실적 검증").Activate
Cells(row_num2, 1) = "사번"
Cells(row_num2, 2) = "지점"
Cells(row_num2, 3) = "성명"
Cells(row_num2, 4) = "중계실적시트 합계"
Cells(row_num2, 5) = "전체 실적시트 합계"

row_num2 = row_num2 + 1

For i = 1 To cnt + 1:
    Worksheets("중계실적").Activate
    name = Cells(row_num3, 4).Value
    name2 = Cells(row_num3, 3).Value
    name3 = Cells(row_num3, 5).Value
    Worksheets("중계실적 검증").Activate
    Cells(row_num2, 1) = name
    Cells(row_num2, 2) = name2
    Cells(row_num2, 3) = name3
    Cells(row_num2, 4) = Application.WorksheetFunction.SumIf(Worksheets("중계실적").Columns("D"), name, Worksheets("중계실적").Columns("O"))
    Cells(row_num2, 5) = Application.WorksheetFunction.SumIf(Worksheets("3월 전체실적").Columns("D"), name, Worksheets("3월 전체실적").Columns("EP"))
    If Cells(row_num2, 4).Value <> Cells(row_num2, 5).Value Then
        Cells(row_num2, 6) = "이상"
    Else
     Cells(row_num2, 6) = ""
    End If
   
 
    row_num3 = row_num3 + 1
    row_num2 = row_num2 + 1
Next
    
End Sub

