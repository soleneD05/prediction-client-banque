#!/bin/bash

make pipeline
uvicorn app.api:app --host "0.0.0.0"