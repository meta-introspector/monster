(* onlyskills.com - zkERDAProlog Skill Registry in OCaml *)

type prime = int
type shard_id = int

type skill = {
  shard_id: shard_id;
  prime: prime;
  skill_name: string;
  skill_type: string;
  command: string;
  search_capability: string;
  zkperf_hash: string;
}

let monster_primes = [2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 41; 47; 59; 71]

let get_prime shard_id =
  List.nth monster_primes (shard_id mod 15)

let create_skill shard_id name skill_type cmd search_cap hash =
  {
    shard_id;
    prime = get_prime shard_id;
    skill_name = name;
    skill_type;
    command = cmd;
    search_capability = search_cap;
    zkperf_hash = hash;
  }

let skill_to_json skill =
  Printf.sprintf 
    {|{"shard_id":%d,"prime":%d,"skill_name":"%s","skill_type":"%s","command":"%s","search_capability":"%s","zkperf_hash":"%s"}|}
    skill.shard_id
    skill.prime
    skill.skill_name
    skill.skill_type
    skill.command
    skill.search_capability
    skill.zkperf_hash

let skill_to_rdf skill =
  let subject = Printf.sprintf "<https://onlyskills.com/skill/%s>" skill.skill_name in
  Printf.sprintf 
    "%s rdf:type zkerdfa:SearchSkill .\n%s zkerdfa:shardId %d .\n%s zkerdfa:prime %d .\n%s zkerdfa:searchType \"%s\" .\n"
    subject subject skill.shard_id subject skill.prime subject skill.search_capability

let () =
  let expert_system = create_skill 29 "expert_system" "search_explicit_search" 
    "cargo run --release --bin expert_system" "explicit_search" "a3f5b2c1d4e6f7a8" in
  
  Printf.printf "OCaml zkERDAProlog Skill Registry\n";
  Printf.printf "==================================\n\n";
  Printf.printf "JSON:\n%s\n\n" (skill_to_json expert_system);
  Printf.printf "RDF:\n%s\n" (skill_to_rdf expert_system);
  Printf.printf "∞ 71 Shards in OCaml ∞\n"
