from template import compare_models

print("Dang chay compare_models với API THAT...\n")
result = compare_models('Viet Nam co bao nhieu tinh?')
print("=== KET QUA SO SANH MODEL ===")
for key, value in result.items():
    print(f"{key}: {value}")