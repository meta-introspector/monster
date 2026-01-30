% Repo Scanner - Prolog
:- module(repo_scanner, [
    scan_repos/2,
    find_duplicates/2,
    prove_novelty/2
]).

:- use_module(library(filesex)).

% Scan all repos
scan_repos(BaseDir, Scans) :-
    find_git_repos(BaseDir, Repos),
    maplist(scan_repo, Repos, Scans).

% Find git repos
find_git_repos(BaseDir, Repos) :-
    directory_files(BaseDir, Files),
    findall(Repo,
        (member(File, Files),
         atom_concat(BaseDir, '/', Base),
         atom_concat(Base, File, Repo),
         exists_directory(Repo),
         atom_concat(Repo, '/.git', GitDir),
         exists_directory(GitDir)),
        Repos).

% Scan single repo
scan_repo(Repo, Scan) :-
    file_base_name(Repo, Name),
    find_docs(Repo, Docs),
    length(Docs, DocCount),
    hash_to_shard(Name, Shard),
    Scan = [
        repo(Repo),
        name(Name),
        shard(Shard),
        doc_count(DocCount)
    ].

% Find docs
find_docs(Repo, Docs) :-
    findall(Doc,
        (directory_member(Repo, Doc, [recursive(true)]),
         file_name_extension(_, Ext, Doc),
         member(Ext, [md, txt, rst])),
        Docs).

% Hash to shard
hash_to_shard(Name, Shard) :-
    atom_codes(Name, Codes),
    sum_list(Codes, Sum),
    Shard is Sum mod 71.

% Find duplicates in 8M files
find_duplicates(Scans, Duplicates) :-
    findall(Name-Count,
        (member(Scan, Scans),
         member(name(Name), Scan),
         findall(S, (member(S, Scans), member(name(Name), S)), Matches),
         length(Matches, Count),
         Count > 1),
        Duplicates).

% Prove novelty (no duplicates)
prove_novelty(Scans, Proof) :-
    find_duplicates(Scans, Duplicates),
    (Duplicates = [] ->
        Proof = [novel(true), duplicates(0)]
    ; length(Duplicates, Count),
      Proof = [novel(false), duplicates(Count)]).
