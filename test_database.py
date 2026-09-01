from sqlalchemy import inspect

from app.db_connection import Base, engine, SessionLocal

from app.database.user import User
from app.database.analysis import Analysis


def main():
    print("Creating database tables...")

    Base.metadata.create_all(bind=engine)

    print("✅ Tables verified successfully!")

    db = SessionLocal()

    try:
        test_email = "test@criticui.com"

        # Check if test user already exists
        existing_user = (
            db.query(User)
            .filter(User.email == test_email)
            .first()
        )

        if existing_user:
            print("\n⚠️ Test user already exists.")
            print(f"User ID: {existing_user.id}")

        else:
            user = User(
                email=test_email,
                hashed_password="TEST_HASH_ONLY",
            )

            db.add(user)
            db.commit()
            db.refresh(user)

            print("\n✅ Test user created successfully!")
            print(f"User ID: {user.id}")
            print(f"Email: {user.email}")

        # Verify users table
        users_count = db.query(User).count()

        print(f"\nUsers in database: {users_count}")

    finally:
        db.close()


if __name__ == "__main__":
    main()