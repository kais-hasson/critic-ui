import modal


app = modal.App("critic-ui")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install_from_requirements(
        "requirements.txt"
    )
    .pip_install(
        "psycopg2-binary==2.9.9",
        "pwdlib==0.3.1",
        "argon2-cffi==25.1.0",
    )
    .add_local_dir(
        ".",
        remote_path="/root/CriticUI",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(
        0,
        "/root/CriticUI"
    )

    from app.api.main import app

    return app