#!/bin/bash

# Function: Show usage help
function usage() {
    echo "Usage: $0 --plus <plus_strand_bw> --minus <minus_strand_bw> --ref <reference_fasta> --type <task_type> -o <output_dir> [--op <output_prefix>]"
    echo
    echo "Required arguments:"
    echo "  --plus     Path to plus strand bigWig file"
    echo "  --minus    Path to minus strand bigWig file"
    echo "  --ref      Reference genome in FASTA format"
    echo "  --type     Subtask type: HistoneModification | FunctionalElement | GeneExpression | TFBS"
    echo "  -o         Output directory"
    echo
    echo "Optional arguments:"
    echo "  --op       Output filename prefix"
    echo "  -h, --help Display this help message and exit"
    exit 1
}

# Default value
OP=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plus)
            PLUS="$2"
            shift 2
            ;;
        --minus)
            MINUS="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        --type)
            TYPE="$2"
            shift 2
            ;;
        -o)
            OUTPUT="$2"
            shift 2
            ;;
        --op)
            OP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "$PLUS" || -z "$MINUS" || -z "$REF" || -z "$TYPE" || -z "$OUTPUT" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Build prediction command
CMD="python predict.py --plus \"$PLUS\" --minus \"$MINUS\" --ref \"$REF\" --type \"$TYPE\" -o \"$OUTPUT\""
if [[ -n "$OP" ]]; then
    CMD="$CMD --op \"$OP\""
fi

# Run the command
echo "Executing:"
echo "$CMD"
eval "$CMD"
