# EduBuddy shell shortcut.
#
# Install once:
#     echo 'source ~/edubuddy/edubuddy/edubuddy.sh' >> ~/.bashrc
#     source ~/.bashrc
#
# Then from anywhere:
#     edubuddy              browser UI          (http://127.0.0.1:7860)
#     edubuddy deploy       same, with question packs OFF  (deploy behaviour)
#     edubuddy app          desktop window
#     edubuddy pack         build the question pack
#     edubuddy test "..."   what a question would match in the pack
#     edubuddy library      validate library_questions.json before a demo
#     edubuddy compare      deploy vs dynamic, writes a table to results/
#     edubuddy bench ...    benchmark.py, any arguments
#     edubuddy eval         full 34-question validation run
#     edubuddy diagnose     retrieval inspector, no model loaded
#     edubuddy books        what is indexed
#     edubuddy results      list what has been recorded
#     edubuddy cd           just go there with the venv active
#
# There is only one installation. 'deploy' and 'dynamic' are the same code;
# the only difference is whether the question pack is consulted before the
# model, which is a single flag. Keeping them as one install is deliberate —
# two copies would drift, and then any comparison between them would be
# measuring the drift rather than the packs.
#
# Change this if the project lives somewhere else.
EDUBUDDY_HOME="${EDUBUDDY_HOME:-$HOME/edubuddy/edubuddy}"

edubuddy() {
    if [ ! -d "$EDUBUDDY_HOME" ]; then
        echo "EduBuddy not found at $EDUBUDDY_HOME" >&2
        echo "Set EDUBUDDY_HOME to the right path and try again." >&2
        return 1
    fi

    # A subshell would lose the activated venv on 'edubuddy cd', so change
    # directory in the caller and come back only where it makes sense.
    cd "$EDUBUDDY_HOME" || return 1

    if [ -f .venv/bin/activate ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo "No virtual environment here. Run ./setup.sh first." >&2
        return 1
    fi

    local cmd="${1:-web}"
    shift 2>/dev/null

    case "$cmd" in
        web)      python web.py "$@" ;;
        # Deploy behaviour: identical code, question packs switched off. Useful
        # for showing the before/after side by side without a second install.
        deploy)   echo "[packs OFF — deploy behaviour]"
                  EDUBUDDY_QA=0 python web.py "$@" ;;
        app)      python app.py "$@" ;;
        pack)     python qa_pack.py --build "$@" ;;
        test)     python qa_pack.py --test "$@" ;;
        library)  python library.py --validate "$@" ;;
        compare)  python benchmark.py --compare-pack "$@" ;;
        bench)    python benchmark.py "$@" ;;
        eval)     python evaluate.py --no-tts "$@" ;;
        diagnose) python diagnose.py "$@" ;;
        books)    python ingest.py --list ;;
        ingest)   python ingest.py --all ;;
        verify)   python verify.py ;;
        results)  ls -lt results/ 2>/dev/null | head -20 ;;
        cd)       echo "$PWD  (venv active)" ;;
        -h|--help|help)
            sed -n '3,18p' "$EDUBUDDY_HOME/edubuddy.sh" | sed 's/^# \{0,1\}//'
            ;;
        *)
            echo "Unknown: $cmd" >&2
            echo "Try: edubuddy help" >&2
            return 1
            ;;
    esac
}

# Tab-completion for the subcommands.
_edubuddy_complete() {
    local words="web deploy app pack test library compare bench eval diagnose books ingest verify results cd help"
    COMPREPLY=($(compgen -W "$words" -- "${COMP_WORDS[1]}"))
}
complete -F _edubuddy_complete edubuddy
