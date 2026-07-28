# grant-copilot — runtime image.
#
# Carries the painful native stack so nobody has to install it by hand:
#   libspatialite  the gazetteer is a SpatiaLite DB; the extension is
#                  loaded at connect time and is the #1 fresh-install
#                  failure on both macOS and Linux
#   swig           build dependency pulled in by parts of the stack
#   faiss-cpu      ships wheels, but pairs badly with mismatched numpy
#
# Deliberately does NOT carry:
#   the ~8 GB data layer — bind-mounted, hydrated once on the host, and
#   far too large to bake into an image that changes with every commit
#
#   the repo itself, in dev use — bind-mounted, because the prompt lab
#   WRITES to prompts/v1/synthesizer.yaml and config/presentation.yaml.
#   Baked-in code would mean every student edit vanished on restart.
#
# The COPY below exists so the image is self-contained enough to run
# without a mount (CI, a quick demo); docker-compose.yml overlays the
# live repo on top for normal use.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- native dependencies ---------------------------------------------
# libsqlite3-mod-spatialite provides mod_spatialite.so, which
# scripts/chatbot/gazetteer_db.py loads via SQLite's extension API.
# No build-essential: every dependency in requirements.txt ships a
# manylinux wheel, so nothing compiles from source. Including it added
# ~400 MB (1.08 GB -> 682 MB) for no benefit. If a future dependency
# does need to compile, add it back here rather than wondering why the
# build broke.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-mod-spatialite \
        libspatialite7 \
        libgeos-c1v5 \
        libproj25 \
        swig \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# gazetteer_db.py calls conn.load_extension("mod_spatialite") with a bare
# name, which SQLite resolves through the normal loader search path. The
# apt package puts it there, so no explicit path is needed — and leaving
# it unset keeps this image working on arm64 (Apple Silicon) as well as
# x86_64, where the library directory differs.

WORKDIR /app

# Dependencies first so a code change doesn't re-install the world.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Non-root. The bind-mounted repo must be writable by this uid for the
# prompt lab to save edits — compose passes the host uid through.
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# 0.0.0.0 so the port mapping reaches it from the host.
CMD ["python", "run_app.py", "--host", "0.0.0.0", "--port", "8000"]
