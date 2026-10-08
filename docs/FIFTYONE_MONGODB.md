# FiftyOne with local MongoDB

FiftyOne stores datasets in MongoDB. Use a local Docker container for inspection; keep MongoDB bound to loopback unless you have deliberately configured authenticated remote access.

## Storage location

MongoDB's WiredTiger database must live on a Linux filesystem that supports its locking and file operations. This repository currently resides on a `fuseblk`/NTFS mount, where MongoDB fails with `Operation not permitted` when opening WiredTiger files.

Keep the actual database under Linux-native storage, and expose the requested repository path as a symlink:

```bash
mkdir -p "$HOME/.local/share/waymo-lidar-transformer/fiftyone/mongodb"
mkdir -p data/fiftyone
ln -s "$HOME/.local/share/waymo-lidar-transformer/fiftyone/mongodb" data/fiftyone/mongodb
```

On this machine, `data/fiftyone/mongodb` resolves to `/home/cheeko/.local/share/waymo-lidar-transformer/fiftyone/mongodb` on ext4. The actual files are outside the FUSE-mounted repository; the in-repo path is the stable project-facing pointer. `data/` is gitignored.

## Start MongoDB

The container binds directly to the Linux-native path. This avoids Docker daemon symlink-resolution differences and ensures MongoDB doesn't use the FUSE filesystem:

```bash
docker run -d \
  --name fiftyone-mongodb \
  --restart unless-stopped \
  -p 127.0.0.1:27017:27017 \
  -v "$HOME/.local/share/waymo-lidar-transformer/fiftyone/mongodb:/data/db" \
  mongo:7.0 --bind_ip_all
```

Check that it is running:

```bash
docker ps --filter name=fiftyone-mongodb
docker logs --tail 30 fiftyone-mongodb
```

Stop/start it without deleting the database:

```bash
docker stop fiftyone-mongodb
docker start fiftyone-mongodb
```

## Configure the project

Create `.env` from the checked-in template and install the project dependencies:

```bash
cp .env.example .env
uv sync --all-groups
```

Set `FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27017` in `.env`. `.env` is gitignored. The shared FiftyOne loader calls `load_dotenv()` before importing FiftyOne, so the setting is applied before its database connection initializes. This URI is suitable when the Python process and MongoDB container share the same host/network namespace.

Install/update dependencies with `uv sync --all-groups`, then open either dataset:

```bash
uv run --group waymo_eval python scripts/export_waymo_eval.py --fiftyone
uv run python -m scripts.generate_synthetic --fiftyone train
```

For a remote pod, run the CLI on the pod and forward the FiftyOne App port (typically `5151`) through your SSH/VS Code tunnel. Keep MongoDB bound to pod-local loopback; do not expose port `27017` publicly.

If the generator or FiftyOne App runs in a different container or on a remote pod, `127.0.0.1` refers to that environment, not the MongoDB host. Configure a reachable private MongoDB address and protect it with authentication/TLS; do not expose an unauthenticated database port publicly.

To remove the container but preserve its data, run `docker rm -f fiftyone-mongodb`. The previous `fiftyone-mongodb-data` Docker volume was retained as a migration backup; do not remove it until you have confirmed the new container contains the datasets you need. The one-time FUSE copy is retained at `data/fiftyone/mongodb-fuse-backup` and is not used by the running service.