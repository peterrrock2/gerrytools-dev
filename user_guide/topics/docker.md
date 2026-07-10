# Docker setup for MGRP

The MGRP runners launch ensemble engines inside a container. GerryTools controls that container
through the Docker SDK, mounts your input and output directories, and translates the Python run
objects into engine-specific commands.

## What must be installed

You need all three of the following:

1. GerryTools with the `mgrp` extra.
2. A Docker-compatible daemon that is running.
3. The MGRP image expected by your version of GerryTools.

Installing the Python package does not install or start Docker.

## Confirm that Docker is reachable

Run these checks before debugging a runner configuration:

```console
docker version
docker info
```

`docker version` should show both a client and a server. If it shows only the client, start Docker
Desktop or the system Docker service.

Then verify the Python SDK from the project environment:

```console
python -c "import docker; print(docker.from_env().ping())"
```

The expected result is `True`.

## Image selection

The image tag is part of a reproducible run. Record the full repository and tag with the analysis
configuration, and avoid relying on a moving `latest` tag for published work.

List local images with:

```console
docker image ls
```

Pull the tag required by your project before a long run:

```console
docker pull mgggdev/replicate:TAG
```

Replace `TAG` with the version documented by the project or release you are using.

## File sharing and mounts

Runner configuration contains host paths that Docker mounts into the container. A path may exist
in Python but still be unavailable to Docker Desktop if its parent directory is not shared.

Use absolute paths when diagnosing a mount problem. On macOS and Windows, confirm that the project
directory is covered by Docker Desktop's file-sharing settings. Avoid placing the project on a
network-mounted or cloud-synchronized directory until a small local run succeeds.

## Linux permissions

On Linux, permission errors usually mean the current user cannot access the Docker socket. Follow
Docker's documented post-install steps for your system. Logging out and back in may be required
after group membership changes.

Do not solve a persistent permissions problem by running an entire analysis as root. Container
outputs may then be owned by root, which creates a second set of file-access problems.

## Resource planning

Containers share the host's CPU, memory, and disk. Before a production run:

- make sure the output filesystem has enough free space;
- confirm Docker Desktop's memory and CPU limits;
- run a small number of steps using the exact same mounts and columns;
- inspect the log and output files before increasing the run size.

Continue with {doc}`../user/mgrp` for runner-specific configuration.
