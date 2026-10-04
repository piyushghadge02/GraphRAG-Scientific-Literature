# Knowledge Bytes — `lib/vis-9.1.2/`

---

### Byte 1: This Folder Vendors the vis-network Runtime
**Builds on:** None — starting point

**In plain terms:**
`vis-network.min.js` is the bundled/minified vis-network JavaScript library used to render and interact with network graphs. The accompanying `vis-network.css` provides the library's browser-side styling.

**The code:**
```javascript
t.DataSet=Kx;
t.DataView=$x;
t.Network=xT;
t.data=Jx;
t.network=TT;
```

The file is third-party bundled code rather than project-specific GraphRAG logic.

---

### Byte 2: `Network` Is the Graph Rendering API
**Builds on:** Byte 1

**In plain terms:**
The bundled library exports `Network`, which is the main API for creating and controlling a rendered graph. The project-specific bindings call methods such as `selectNodes()` and `getConnectedNodes()` on this object.

**The code:**
```javascript
t.Network=xT;
t.network=TT;
```

The implementation is minified, so the useful way to understand it at this level is through its public API rather than individual minified functions.

---

### Byte 3: `DataSet` Holds Graph Data
**Builds on:** Byte 2

**In plain terms:**
vis-network provides `DataSet` as a data container for graph items. The project's `utils.js` uses the application's `nodes` and `edges` objects through methods such as `get()` and `update()`.

**The code:**
```javascript
t.DataSet=Kx;
t.DataView=$x;
```

This explains why the custom bindings can read all nodes, modify their display properties, and write the changes back.

---

### Byte 4: `DataView` Provides a Data Abstraction
**Builds on:** Byte 3

**In plain terms:**
The bundled library also exports `DataView`. It is part of vis's data layer and provides another way to work with graph data without changing the main rendering API.

**The code:**
```javascript
t.DataView=$x;
```

The custom GraphRAG binding file shown in this `lib` folder does not directly call `DataView`, so its presence is primarily a library capability.

---

### Byte 5: Graph Interaction Is More Than Rendering
**Builds on:** Byte 2

**In plain terms:**
The bundled runtime contains functionality for interaction, physics, configuration, manipulation, and navigation controls. These capabilities explain the behaviors that a graph UI can expose beyond simply drawing nodes and edges.

**The code:**
```text
interaction
physics
configure
manipulation
navigationButtons
```

These names are present in the vendored runtime; the project-specific `utils.js` uses a smaller subset of the available API.

---

### Byte 6: The CSS Is the Visual Companion to the Runtime
**Builds on:** Byte 1

**In plain terms:**
`vis-network.css` is a stylesheet asset shipped alongside the JavaScript library. It provides CSS rules needed by the vis-network UI components.

**The code:**
```text
lib/vis-9.1.2/
├── vis-network.min.js
└── vis-network.css
```

It contains presentation rules rather than GraphRAG retrieval or graph-construction logic.

---

### Byte 7: The Library Should Be Treated as a Dependency Boundary
**Builds on:** Bytes 1–6

**In plain terms:**
Because this folder contains a bundled third-party library, the GraphRAG-specific behavior should be learned primarily from `bindings/utils.js`. The vendored files provide the runtime capabilities that the bindings consume.

**The code:**
```javascript
network.getConnectedNodes(selectedNode);
network.selectNodes(nodes);
nodes.get(...);
nodes.update(...);
```

These calls are the practical boundary between project code and the third-party graph library.

---

## PUTTING IT TOGETHER

`vis-9.1.2` supplies the graph-rendering runtime and its CSS. `Network` controls the rendered graph, while `DataSet` provides the node/edge data abstraction used by the project's bindings. The custom `bindings/utils.js` sits on top of this API and implements the application's highlighting and filtering behavior. The minified library itself is best treated as a dependency boundary rather than as GraphRAG business logic.
