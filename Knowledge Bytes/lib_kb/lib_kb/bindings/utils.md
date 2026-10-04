# Knowledge Bytes — `lib/bindings/utils.js`

---

### Byte 1: `utils.js` Provides Graph Interaction Helpers
**Builds on:** None — starting point

**In plain terms:**
This file contains browser-side helper functions for interacting with a vis-network graph. Its main responsibilities are highlighting a selected node's neighborhood, filtering visible nodes, selecting nodes, and applying property-based filters.

**The code:**
```javascript
function neighbourhoodHighlight(params) { ... }

function filterHighlight(params) { ... }

function selectNode(nodes) { ... }

function selectNodes(nodes) { ... }

function highlightFilter(filter) { ... }
```

The functions rely on graph objects such as `network`, `nodes`, `edges`, and `nodeColors` supplied by the surrounding application.

---

### Byte 2: Neighborhood Highlighting Starts from the Selected Node
**Builds on:** Byte 1

**In plain terms:**
`neighbourhoodHighlight()` changes the graph's visual emphasis when a node is selected. It makes unrelated nodes visually subdued while keeping the selected node and nearby nodes readable.

**The code:**
```javascript
if (params.nodes.length > 0) {
  highlightActive = true;
  var selectedNode = params.nodes[0];
  var degrees = 2;
  ...
}
```

The function treats the first selected node as the center of the neighborhood operation.

---

### Byte 3: Unrelated Nodes Are De-emphasized
**Builds on:** Byte 2

**In plain terms:**
Before emphasizing the neighborhood, the function changes every node's color and temporarily removes visible labels. It stores the original label so it can be restored later.

**The code:**
```javascript
for (let nodeId in allNodes) {
  allNodes[nodeId].color =
    "rgba(200,200,200,0.5)";

  if (allNodes[nodeId].hiddenLabel === undefined) {
    allNodes[nodeId].hiddenLabel =
      allNodes[nodeId].label;
    allNodes[nodeId].label = undefined;
  }
}
```

This is a presentation technique: the graph is not structurally modified; only node display properties are changed.

---

### Byte 4: First-Degree Connections Are Restored
**Builds on:** Byte 3

**In plain terms:**
The selected node's directly connected neighbors are found through the vis-network API. Their original colors and labels are restored so the immediate neighborhood stands out.

**The code:**
```javascript
var connectedNodes =
  network.getConnectedNodes(selectedNode);

for (i = 0; i < connectedNodes.length; i++) {
  allNodes[connectedNodes[i]].color =
    nodeColors[connectedNodes[i]];
}
```

The `nodeColors` object is the source of the original per-node colors.

---

### Byte 5: The Code Also Looks One More Hop Away
**Builds on:** Byte 4

**In plain terms:**
The function expands through each directly connected node to collect another layer of connected nodes. These nodes receive a softer visual treatment than the selected node and its immediate neighbors.

**The code:**
```javascript
var allConnectedNodes = [];

for (i = 1; i < degrees; i++) {
  for (j = 0; j < connectedNodes.length; j++) {
    allConnectedNodes = allConnectedNodes.concat(
      network.getConnectedNodes(connectedNodes[j])
    );
  }
}
```

The configured `degrees = 2` makes the function show a small local neighborhood instead of attempting to emphasize the whole graph.

---

### Byte 6: Node Changes Are Pushed Back into the DataSet
**Builds on:** Bytes 3–5

**In plain terms:**
The function first works with an object representation of all nodes, then converts the changed nodes into an array and sends them back through `nodes.update()`.

**The code:**
```javascript
var updateArray = [];

for (let nodeId in allNodes) {
  if (allNodes.hasOwnProperty(nodeId)) {
    updateArray.push(allNodes[nodeId]);
  }
}

nodes.update(updateArray);
```

This is how visual property changes become visible in the rendered graph.

---

### Byte 7: Deselecting Resets the Neighborhood Highlight
**Builds on:** Byte 2

**In plain terms:**
When no node is selected but highlighting was previously active, the function restores every node's original color and any hidden labels.

**The code:**
```javascript
} else if (highlightActive === true) {
  for (let nodeId in allNodes) {
    allNodes[nodeId].color = nodeColors[nodeId];

    if (allNodes[nodeId].hiddenLabel !== undefined) {
      allNodes[nodeId].label =
        allNodes[nodeId].hiddenLabel;
      allNodes[nodeId].hiddenLabel = undefined;
    }
  }

  highlightActive = false;
}
```

The `highlightActive` flag prevents the reset logic from running unnecessarily.

---

### Byte 8: `filterHighlight()` Uses Visibility Instead of Color
**Builds on:** Byte 1

**In plain terms:**
Filtering is different from neighborhood highlighting. Instead of merely dimming unrelated nodes, `filterHighlight()` hides them and temporarily removes their labels.

**The code:**
```javascript
for (let nodeId in allNodes) {
  allNodes[nodeId].hidden = true;

  if (allNodes[nodeId].savedLabel === undefined) {
    allNodes[nodeId].savedLabel =
      allNodes[nodeId].label;
    allNodes[nodeId].label = undefined;
  }
}
```

Only the selected nodes are then made visible again.

---

### Byte 9: Filter Reset Restores All Nodes
**Builds on:** Byte 8

**In plain terms:**
When the selection becomes empty after filtering was active, every node is made visible and saved labels are restored.

**The code:**
```javascript
} else if (filterActive === true) {
  for (let nodeId in allNodes) {
    allNodes[nodeId].hidden = false;

    if (allNodes[nodeId].savedLabel !== undefined) {
      allNodes[nodeId].label =
        allNodes[nodeId].savedLabel;
      allNodes[nodeId].savedLabel = undefined;
    }
  }

  filterActive = false;
}
```

The state flag again distinguishes an actual reset from an ordinary empty selection.

---

### Byte 10: `selectNode()` Selects and Highlights
**Builds on:** Bytes 2–7

**In plain terms:**
`selectNode()` is a convenience wrapper for a single-node interaction. It tells the network to select the nodes and immediately invokes neighborhood highlighting.

**The code:**
```javascript
function selectNode(nodes) {
  network.selectNodes(nodes);
  neighbourhoodHighlight({ nodes: nodes });
  return nodes;
}
```

It combines selection and presentation behavior into one reusable operation.

---

### Byte 11: `selectNodes()` Selects and Filters
**Builds on:** Bytes 8–9

**In plain terms:**
`selectNodes()` is the multi-node counterpart used for filtering behavior. It selects the supplied nodes and then hides everything outside that selection.

**The code:**
```javascript
function selectNodes(nodes) {
  network.selectNodes(nodes);
  filterHighlight({nodes: nodes});
  return nodes;
}
```

The distinction is intentional: one helper creates a neighborhood view, while the other creates a filtered view.

---

### Byte 12: `highlightFilter()` Converts Property Matches into Node Selection
**Builds on:** Byte 11

**In plain terms:**
`highlightFilter()` accepts a filter describing a property and allowed values. For node filters, it finds nodes whose property matches one of the requested values and passes their IDs to `selectNodes()`.

**The code:**
```javascript
if (filter['item'] === 'node') {
  let allNodes =
    nodes.get({ returnType: "Object" });

  for (let nodeId in allNodes) {
    if (
      allNodes[nodeId][selectedProp] &&
      filter['value'].includes(
        allNodes[nodeId][selectedProp].toString()
      )
    ) {
      selectedNodes.push(nodeId);
    }
  }
}
```

The function is therefore a bridge between declarative filter criteria and the graph's visual selection behavior.

---

### Byte 13: Edge Filters Select the Nodes at Both Ends
**Builds on:** Byte 12

**In plain terms:**
For an edge filter, the code checks an edge property instead of a node property. When an edge matches, both endpoint node IDs are collected and then selected.

**The code:**
```javascript
if (filter['item'] === 'edge') {
  let allEdges =
    edges.get({returnType: 'object'});

  for (let edge in allEdges) {
    if (
      allEdges[edge][selectedProp] &&
      filter['value'].includes(
        allEdges[edge][selectedProp].toString()
      )
    ) {
      selectedNodes.push(allEdges[edge]['from']);
      selectedNodes.push(allEdges[edge]['to']);
    }
  }
}
```

The UI therefore represents an edge-property match by highlighting the nodes connected by that edge.

---

### Byte 14: Important State and Global Dependencies
**Builds on:** All previous bytes

**In plain terms:**
This file does not create the graph itself. It assumes `nodes`, `edges`, `network`, `nodeColors`, `highlightActive`, and `filterActive` already exist in the surrounding browser environment.

**The code:**
```javascript
network.selectNodes(nodes);
nodes.update(updateArray);
edges.get({ returnType: 'object' });
```

That coupling is the main gotcha when moving or testing this file independently: the helpers require the graph application's shared state.

---

## PUTTING IT TOGETHER

`utils.js` is a presentation/control layer around the vis-network graph. Selection can produce either a neighborhood highlight or a strict filtered view, while property filters first find matching graph objects and then reuse the selection helpers. The functions modify node display properties and push those changes through the vis `DataSet`. The file therefore sits between user interaction and the underlying graph data rather than constructing the graph itself.
