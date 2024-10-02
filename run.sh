#!/bin/bash
# using unbuffer because of the pipe to tee 
# https://superuser.com/questions/352697/preserve-colors-while-piping-to-tee
go build ./cmd/main.go && unbuffer ./main "$@" | tee output/$(date +%s).txt
