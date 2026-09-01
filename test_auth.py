from app.auth.security import hash_password, verify_password


def main():
    password = "MySecurePassword123!"

    print("Original password:")
    print(password)

    hashed = hash_password(password)

    print("\nHashed password:")
    print(hashed)

    print("\nCorrect password verification:")
    print(verify_password(password, hashed))

    print("\nWrong password verification:")
    print(verify_password("WrongPassword123!", hashed))


if __name__ == "__main__":
    main()