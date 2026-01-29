#!/usr/bin/env gap
# Export Monster group data from GAP to JSON

# Load required packages
LoadPackage("atlasrep");
LoadPackage("ctbllib");

Print("🔢 Exporting Monster Group Data\n");
Print("================================\n\n");

# Load Monster group
Print("📥 Loading Monster group...\n");
M := AtlasGroup("M");

if M = fail then
    Print("❌ Failed to load Monster group\n");
    QUIT_GAP(1);
fi;

Print("✅ Monster group loaded\n\n");

# Get basic properties
Print("📊 Computing properties...\n");

data := rec(
    name := "Monster",
    order := Order(M),
    order_string := String(Order(M)),
    is_simple := IsSimple(M),
    is_sporadic := true
);

Print("✅ Order: ", data.order_string, "\n");

# Get character table
Print("📊 Loading character table...\n");
ct := CharacterTable("M");

if ct <> fail then
    data.num_conjugacy_classes := NrConjugacyClasses(ct);
    data.num_characters := Length(Irr(ct));
    Print("✅ Conjugacy classes: ", data.num_conjugacy_classes, "\n");
    Print("✅ Characters: ", data.num_characters, "\n");
fi;

# Export to JSON
Print("\n💾 Exporting to JSON...\n");

json_str := Concatenation(
    "{\n",
    "  \"name\": \"", data.name, "\",\n",
    "  \"order\": \"", data.order_string, "\",\n",
    "  \"is_simple\": ", String(data.is_simple), ",\n",
    "  \"is_sporadic\": ", String(data.is_sporadic), ",\n",
    "  \"num_conjugacy_classes\": ", String(data.num_conjugacy_classes), ",\n",
    "  \"num_characters\": ", String(data.num_characters), "\n",
    "}\n"
);

# Write to file
output_file := "monster_gap_data.json";
PrintTo(output_file, json_str);

Print("✅ Exported to ", output_file, "\n");
Print("\n🎯 Monster group data exported successfully!\n");

QUIT_GAP(0);
