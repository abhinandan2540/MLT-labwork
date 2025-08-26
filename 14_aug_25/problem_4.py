
# verification regex
email = str(input("enter email: "))
password = str(input("enter password: "))

if '@' in email and len(password) >= 8:
    print('valid email and password')
else:
    print("some params missing")
