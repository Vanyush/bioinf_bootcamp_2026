import SwiftUI
import UniformTypeIdentifiers
import Foundation
import AppKit
internal import Combine

struct SequenceRecord: Identifiable, Hashable {
    let id = UUID()
    let header: String
    let sequence: String
    let extractedAgeBP: Double?

    var displayName: String {
        header.trimmingCharacters(in: CharacterSet(charactersIn: ">"))
    }
}

struct HaplotypeCluster: Identifiable, Hashable {
    let id = UUID()
    let index: Int
    let label: String
    let sequence: String
    let memberIndices: [Int]
    let memberHeaders: [String]
    let observedAges: [Double]

    var count: Int { memberIndices.count }
    var observedAgeBP: Double? { Self.median(observedAges) }

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

struct NetworkEdge: Identifiable, Hashable {
    let id = UUID()
    let a: Int
    let b: Int
    let distance: Int

    var weight: Double {
        1.0 / Double(distance + 1)
    }
}

struct NetworkNode: Identifiable, Hashable {
    let id = UUID()
    let index: Int
    let label: String
    let consensusSequence: String
    let count: Int
    let memberHeaders: [String]
    let observedAgeBP: Double?
    let inferredAgeBP: Double?
    let confidence: Double
    let isObservedAge: Bool
}

struct TempNetResult {
    let records: [SequenceRecord]
    let distances: [[Int]]
    let networkEdges: [NetworkEdge]
    let nodes: [NetworkNode]
    let warnings: [String]
}

// MARK: - FASTA парсинг | FASTA parsing

enum FASTAParser {
    static func parse(_ text: String) -> [SequenceRecord] {
        let lines = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: .newlines)

        var records: [SequenceRecord] = []
        var currentHeader: String?
        var sequenceParts: [String] = []

        func flush() {
            guard let header = currentHeader else { return }
            let rawSeq = sequenceParts.joined()
                .uppercased()
                .replacingOccurrences(of: " ", with: "")
            let age = extractAge(from: header)
            records.append(SequenceRecord(header: header, sequence: rawSeq, extractedAgeBP: age))
        }

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed.hasPrefix(">") {
                flush()
                currentHeader = trimmed
                sequenceParts = []
            } else {
                sequenceParts.append(trimmed)
            }
        }
        flush()

        return records
    }

    static func extractAge(from header: String) -> Double? {
        let body = header.trimmingCharacters(in: CharacterSet(charactersIn: ">"))
        let parts = body.split(separator: "_")
        if let last = parts.last, let value = Double(last.replacingOccurrences(of: ",", with: ".")), value > 0 {
            return value
        }

        let pattern = #"(\d{3,6}(?:[\.,]\d+)?)\s*$"#
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let range = NSRange(body.startIndex..<body.endIndex, in: body)
            if let match = regex.firstMatch(in: body, range: range), match.numberOfRanges >= 2,
               let r = Range(match.range(at: 1), in: body) {
                let token = String(body[r]).replacingOccurrences(of: ",", with: ".")
                return Double(token)
            }
        }
        return nil
    }
}

// MARK: - Математическое ядро | Math core

struct TempNetAlgorithm {
    static func buildHaplotypeClusters(from records: [SequenceRecord]) -> [HaplotypeCluster] {
        let grouped = Dictionary(grouping: records.enumerated(), by: { $0.element.sequence })

        var clusters: [HaplotypeCluster] = []
        for (idx, item) in grouped.sorted(by: { $0.key < $1.key }).enumerated() {
            let sequence = item.key
            let group = item.value
            let memberIndices = group.map { $0.offset }.sorted()
            let memberHeaders = group.map { $0.element.displayName }
            let ages = group.compactMap { $0.element.extractedAgeBP }
            let label = "HG\(idx + 1)"
            clusters.append(
                HaplotypeCluster(
                    index: idx,
                    label: label,
                    sequence: sequence,
                    memberIndices: memberIndices,
                    memberHeaders: memberHeaders,
                    observedAges: ages
                )
            )
        }
        return clusters
    }

    static func computeDistances(_ records: [SequenceRecord]) -> [[Int]] {
        let n = records.count
        var matrix = Array(repeating: Array(repeating: 0, count: n), count: n)
        guard n > 1 else { return matrix }

        let seqs = records.map { Array($0.sequence) }

        for i in 0..<n {
            for j in (i + 1)..<n {
                var diff = 0
                var comparable = 0
                let a = seqs[i]
                let b = seqs[j]
                let m = min(a.count, b.count)
                for k in 0..<m {
                    let x = a[k]
                    let y = b[k]
                    if x == "-" || y == "-" || x == "N" || y == "N" { continue }
                    comparable += 1
                    if x != y { diff += 1 }
                }
                if comparable == 0 {
                    diff = abs(a.count - b.count)
                }
                matrix[i][j] = diff
                matrix[j][i] = diff
            }
        }
        return matrix
    }

    static func buildHaplotypeNetwork(from distances: [[Int]]) -> [NetworkEdge] {
        let n = distances.count
        guard n > 1 else { return [] }

        var edges: [NetworkEdge] = []
        var seen = Set<String>()

        func addEdge(_ a: Int, _ b: Int) {
            let x = min(a, b)
            let y = max(a, b)
            let key = "\(x)-\(y)"
            guard seen.insert(key).inserted else { return }
            edges.append(NetworkEdge(a: x, b: y, distance: distances[x][y]))
        }

        var nonZero: [Int] = []
        for i in 0..<n {
            for j in (i + 1)..<n {
                let d = distances[i][j]
                if d > 0 { nonZero.append(d) }
            }
        }
        let sorted = nonZero.sorted()
        let q1 = sorted.isEmpty ? 1 : sorted[max(0, sorted.count / 4)]
        let threshold = max(1, q1)

        for i in 0..<n {
            let rowMin = (0..<n)
                .filter { $0 != i }
                .map { distances[i][$0] }
                .filter { $0 > 0 }
                .min() ?? .max

            for j in (i + 1)..<n {
                let d = distances[i][j]
                if d > 0 && (d <= threshold || d == rowMin) {
                    addEdge(i, j)
                }
            }
        }

        var inTree = Array(repeating: false, count: n)
        var bestDist = Array(repeating: Int.max, count: n)
        var parent = Array(repeating: -1, count: n)
        bestDist[0] = 0

        for _ in 0..<n {
            var u = -1
            var minVal = Int.max
            for i in 0..<n where !inTree[i] && bestDist[i] < minVal {
                minVal = bestDist[i]
                u = i
            }
            guard u >= 0 else { break }
            inTree[u] = true

            for v in 0..<n where !inTree[v] {
                let d = distances[u][v]
                if d < bestDist[v] {
                    bestDist[v] = d
                    parent[v] = u
                }
            }
        }

        for v in 1..<n where parent[v] >= 0 {
            addEdge(parent[v], v)
        }

        return edges.sorted {
            if $0.distance == $1.distance {
                return ($0.a, $0.b) < ($1.a, $1.b)
            }
            return $0.distance < $1.distance
        }
    }

    static func inferAges(
        clusters: [HaplotypeCluster],
        distances: [[Int]],
        networkEdges: [NetworkEdge],
        warnings: inout [String]
    ) -> [NetworkNode] {
        let n = clusters.count
        guard n > 0 else { return [] }

        var adjacency: [[(Int, Double)]] = Array(repeating: [], count: n)
        for edge in networkEdges {
            let w = edge.weight
            adjacency[edge.a].append((edge.b, w))
            adjacency[edge.b].append((edge.a, w))
        }

        let observed: Set<Int> = Set(clusters.enumerated().compactMap { idx, cluster in
            cluster.observedAgeBP != nil ? idx : nil
        })

        var ages = Array(repeating: Double.nan, count: n)
        for (idx, cluster) in clusters.enumerated() {
            if let age = cluster.observedAgeBP {
                ages[idx] = age
            }
        }

        if observed.isEmpty {
            warnings.append("В FASTA не найдено ни одной датированной последовательности. Возраст будет приблизительно привязан к генетическому сходству")
            let root = 0
            let scale = 1000.0
            for i in 0..<n {
                ages[i] = Double(distances[root][i]) * scale
            }
        } else {
            for i in 0..<n where !observed.contains(i) {
                let vals = observed.compactMap { j -> (Double, Double)? in
                    let d = max(1, distances[i][j])
                    guard let age = clusters[j].observedAgeBP else { return nil }
                    return (age, 1.0 / Double(d * d))
                }
                if vals.isEmpty {
                    ages[i] = median(clusters.compactMap { $0.observedAgeBP }) ?? 0.0
                } else {
                    let sumW = vals.reduce(0.0) { $0 + $1.1 }
                    let sum = vals.reduce(0.0) { $0 + $1.0 * $1.1 }
                    ages[i] = sum / max(sumW, 1e-9)
                }
            }

            let iterations = 250
            for _ in 0..<iterations {
                var next = ages
                for i in 0..<n where !observed.contains(i) {
                    let neighbors = adjacency[i]
                    guard !neighbors.isEmpty else { continue }
                    let sumW = neighbors.reduce(0.0) { $0 + $1.1 }
                    let sum = neighbors.reduce(0.0) { $0 + $1.1 * ages[$1.0] }
                    let graphValue = sum / max(sumW, 1e-9)
                    let prior = weightedAgePrior(
                        for: i,
                        clusters: clusters,
                        distances: distances,
                        observed: observed
                    )
                    next[i] = 0.70 * graphValue + 0.30 * prior
                }
                ages = next
            }
        }

        let confidences = (0..<n).map { i -> Double in
            if observed.contains(i) { return 1.0 }
            let nearest = observed.map { Double(distances[i][$0]) }.min() ?? 0.0
            let localSpread = adjacency[i].map { abs(ages[$0.0] - ages[i]) }.reduce(0.0, +) / max(Double(adjacency[i].count), 1.0)
            let c1 = 1.0 / (1.0 + nearest)
            let c2 = 1.0 / (1.0 + localSpread / 1000.0)
            return max(0.05, min(0.98, 0.55 * c1 + 0.45 * c2))
        }

        return clusters.enumerated().map { idx, cluster in
            NetworkNode(
                index: idx,
                label: cluster.label,
                consensusSequence: cluster.sequence,
                count: cluster.count,
                memberHeaders: cluster.memberHeaders,
                observedAgeBP: cluster.observedAgeBP,
                inferredAgeBP: ages[idx].isFinite ? ages[idx] : nil,
                confidence: confidences[idx],
                isObservedAge: cluster.observedAgeBP != nil
            )
        }
    }

    private static func weightedAgePrior(
        for index: Int,
        clusters: [HaplotypeCluster],
        distances: [[Int]],
        observed: Set<Int>
    ) -> Double {
        let candidates = observed.sorted { distances[index][$0] < distances[index][$1] }
        let nearest = candidates.prefix(5)
        let pairs = nearest.compactMap { j -> (Double, Double)? in
            guard let age = clusters[j].observedAgeBP else { return nil }
            let d = max(1, distances[index][j])
            return (age, 1.0 / Double(d * d))
        }
        if pairs.isEmpty {
            return median(clusters.compactMap { $0.observedAgeBP }) ?? 0.0
        }
        let sumW = pairs.reduce(0.0) { $0 + $1.1 }
        let sum = pairs.reduce(0.0) { $0 + $1.0 * $1.1 }
        return sum / max(sumW, 1e-9)
    }

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[mid]
        }
        return (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

final class TempNetModel: ObservableObject {
    @Published var fastaText: String = ""
    @Published var records: [SequenceRecord] = []
    @Published var result: TempNetResult? = nil
    @Published var warnings: [String] = []
    @Published var statusText: String = "Ожидание загрузки FASTA"
    @Published var selectedFileName: String = ""

    func loadFASTA(text: String, fileName: String = "") {
        fastaText = text
        selectedFileName = fileName
        records = FASTAParser.parse(text)
        analyze()
    }

    func analyze() {
        warnings = []

        guard !records.isEmpty else {
            result = nil
            statusText = "FASTA не содержит последовательностей"
            return
        }

        let lengths = Set(records.map { $0.sequence.count })
        if lengths.count != 1 {
            warnings.append("Последовательности имеют разную длину. Для этого алгоритма желательно выравнивание одной длины")
        }

        let cleaned = records.map { rec in
            SequenceRecord(
                header: rec.header,
                sequence: rec.sequence.uppercased(),
                extractedAgeBP: rec.extractedAgeBP
            )
        }

        let haplotypeClusters = TempNetAlgorithm.buildHaplotypeClusters(from: cleaned)
        if haplotypeClusters.count < cleaned.count {
            warnings.append("Идентичные последовательности были объединены в гаплогруппы")
        }

        let clusterRecords = haplotypeClusters.map {
            SequenceRecord(header: $0.label + "_" + String($0.count), sequence: $0.sequence, extractedAgeBP: $0.observedAgeBP)
        }

        let distances = TempNetAlgorithm.computeDistances(clusterRecords)
        let networkEdges = TempNetAlgorithm.buildHaplotypeNetwork(from: distances)
        let nodes = TempNetAlgorithm.inferAges(
            clusters: haplotypeClusters,
            distances: distances,
            networkEdges: networkEdges,
            warnings: &warnings
        )

        result = TempNetResult(
            records: clusterRecords,
            distances: distances,
            networkEdges: networkEdges,
            nodes: nodes,
            warnings: warnings
        )

        let dated = clusterRecords.compactMap { $0.extractedAgeBP }
        let undated = clusterRecords.count - dated.count
        statusText = "Готово:\n \(cleaned.count) последовательностей\n \(clusterRecords.count) гаплогрупп\n \(dated.count) датированных\n \(undated) без возраста"
    }
}

// MARK: - Графический интерфейс | GUI

struct ContentView: View {
    @StateObject private var model = TempNetModel()
    @State private var isImporterPresented = false
    @State private var selectedNodeIndex: Int? = nil

    var body: some View {
        NavigationSplitView {
            sidebar
        .navigationSplitViewColumnWidth(min: 180, ideal: 190, max: 220)
        } detail: {
            detailView
        }
        .frame(minWidth: 1200, minHeight: 800)
        .onAppear {
            if model.records.isEmpty {
                model.statusText = "Ожидание"
            }
        }
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [UTType(filenameExtension: "fasta") ?? .plainText, .plainText, UTType(filenameExtension: "fa") ?? .plainText],
            allowsMultipleSelection: false
        ) { result in
            do {
                guard let url = try result.get().first else { return }
                let text = try String(contentsOf: url, encoding: .utf8)
                model.loadFASTA(text: text, fileName: url.lastPathComponent)
            } catch {
                model.statusText = "Не удалось открыть файл: \(error.localizedDescription)"
            }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("MMNV")
                .font(.largeTitle.bold())
            Text("Тестовое приложение для хакатона 'биоинф буткемп' \nКоманда: Бригада 9")
                .foregroundStyle(.secondary)

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Button {
                        isImporterPresented = true
                    } label: {
                        Label("Открыть FASTA", systemImage: "folder.badge.plus")
                    }

                    Text(model.selectedFileName.isEmpty ? "Файл не выбран" : model.selectedFileName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            GroupBox("Статус") {
                Text(model.statusText)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }


            Spacer()
        }
        .padding()
    }

    private var detailView: some View {
        Group {
            if let result = model.result {
                HStack(spacing: 0) {
                    VStack(spacing: 0) {
                        headerBar(result: result)
                        Divider()
                        NetworkCanvasView(result: result, selectedNodeIndex: $selectedNodeIndex)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(Color(nsColor: .windowBackgroundColor))
                    }
                    .frame(minWidth: 750)

                    Divider()

                    SequenceTableView(result: result, selectedNodeIndex: $selectedNodeIndex)
                        .frame(minWidth: 380)
                }
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "network")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("Импортируйте FASTA-файл, чтобы построить временную сеть")
                        .font(.title3)
                    Text("Приложение рассчитано на выравненные митохондриальные последовательности ДНК и умеет объединять схожие последовательности в гаплогруппы и прогнозировать возраст недатированных образцов по генетическому сходству")
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 500)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private func headerBar(result: TempNetResult) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Временная сеть гаплогрупп")
                    .font(.headline)
                Text("Голубые узлы — датированные образцы, оранжевые — прогнозные")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                exportCSV(result: result)
            } label: {
                Label("Экспорт CSV", systemImage: "square.and.arrow.up")
            }
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
    }

    private func exportCSV(result: TempNetResult) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.nameFieldStringValue = "MMNV_results.csv"
        panel.canCreateDirectories = true

        if panel.runModal() == .OK, let url = panel.url {
            let sortedNodes = result.nodes.sorted { lhs, rhs in
                let la = lhs.inferredAgeBP ?? -Double.greatestFiniteMagnitude
                let ra = rhs.inferredAgeBP ?? -Double.greatestFiniteMagnitude
                if la == ra { return lhs.index < rhs.index }
                return la > ra
            }

            var csv = "index,label,members,observed_age,inferred_age,confidence,sequence_length\n"
            for node in sortedNodes {
                let observed = node.observedAgeBP.map { String(format: "%.0f", $0) } ?? ""
                let inferred = node.inferredAgeBP.map { String(format: "%.2f", $0) } ?? ""
                let confidence = String(format: "%.3f", node.confidence)
                let row = [
                    String(node.index),
                    csvEscape(node.label),
                    String(node.count),
                    observed,
                    inferred,
                    confidence,
                    String(node.consensusSequence.count)
                ].joined(separator: ",")
                csv += row + "\n"
            }
            do {
                try csv.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                model.statusText = "Не удалось сохранить CSV: \(error.localizedDescription)"
            }
        }
    }

    private func csvEscape(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") {
            return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return value
    }
}

struct SequenceTableView: View {
    let result: TempNetResult
    @Binding var selectedNodeIndex: Int?

    var body: some View {
        let displayNodes = result.nodes.sorted { lhs, rhs in
            let la = lhs.inferredAgeBP ?? -Double.greatestFiniteMagnitude
            let ra = rhs.inferredAgeBP ?? -Double.greatestFiniteMagnitude
            if la == ra { return lhs.index < rhs.index }
            return la > ra
        }

        VStack(alignment: .leading, spacing: 0) {
            Text("Гаплогруппы")
                .font(.headline)
                .padding()

            List(selection: $selectedNodeIndex) {
                ForEach(displayNodes) { node in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(node.label)
                                .lineLimit(2)
                            Spacer()
                            if node.isObservedAge {
                                Text("\(Int(node.observedAgeBP ?? 0)) лет")
                                    .foregroundStyle(.blue)
                            } else if let inferred = node.inferredAgeBP {
                                Text("~\(Int(inferred)) лет")
                                    .foregroundStyle(.orange)
                            }
                        }
                        VStack(alignment: .leading, spacing: 6) {

                            HStack(spacing: 12) {
                                Text("Количество последовательностей: \(node.count)")
                                Text("Confidence: \(String(format: "%.2f", node.confidence))")
                                Text(node.isObservedAge ? "датирован" : "прогноз")
                            }

                            VStack(alignment: .leading, spacing: 2) {

                                Text("Последовательности:")
                                    .font(.caption.bold())

                                ForEach(node.memberHeaders, id: \.self) { seq in
                                    Text(seq)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                    .tag(node.index)
                }
            }
            .listStyle(.plain)
        }
    }
}

struct NetworkCanvasView: View {
    let result: TempNetResult
    @Binding var selectedNodeIndex: Int?

    @State private var zoomLevel: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var lastDragPosition: CGPoint? = nil
    @State private var eventMonitor: Any? = nil
    @State private var canvasSize: CGSize = .zero

    private var ageBounds: (min: Double, max: Double) {
        let ages = result.nodes.compactMap { $0.inferredAgeBP }
        let minA = ages.min() ?? 0
        let maxA = ages.max() ?? 1
        if minA == maxA {
            return (minA - 1, maxA + 1)
        }
        return (minA, maxA)
    }

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                DispatchQueue.main.async {
                    canvasSize = size
                }
                let logicalPositions = layout(size: size)

                func transform(_ point: CGPoint) -> CGPoint {
                    let center = CGPoint(x: size.width / 2, y: size.height / 2)
                    let scaledX = center.x + (point.x - center.x) * zoomLevel + panOffset.width
                    let scaledY = center.y + (point.y - center.y) * zoomLevel + panOffset.height
                    return CGPoint(x: scaledX, y: scaledY)
                }

                for edge in result.networkEdges {
                    guard let p1 = logicalPositions[edge.a], let p2 = logicalPositions[edge.b] else { continue }
                    let tp1 = transform(p1)
                    let tp2 = transform(p2)

                    let isConnectedToSelected = (selectedNodeIndex != nil) && (edge.a == selectedNodeIndex || edge.b == selectedNodeIndex)
                    let lineWidth: CGFloat = isConnectedToSelected ? 3.0 : 1.4
                    let strokeColor: Color = isConnectedToSelected ? .blue : .secondary.opacity(0.35)

                    var path = Path()
                    path.move(to: tp1)
                    let midX = (tp1.x + tp2.x) / 2
                    let curvature: CGFloat = abs(tp1.y - tp2.y) * 0.15 + 20
                    path.addCurve(to: tp2,
                                  control1: CGPoint(x: midX, y: tp1.y - curvature),
                                  control2: CGPoint(x: midX, y: tp2.y + curvature))
                    context.stroke(path, with: .color(strokeColor), lineWidth: lineWidth)
                }

                let axisYLogical = size.height - 34
                let axisYTransformed = transform(CGPoint(x: 0, y: axisYLogical)).y
                var axisPath = Path()
                axisPath.move(to: CGPoint(x: 0, y: axisYTransformed))
                axisPath.addLine(to: CGPoint(x: size.width, y: axisYTransformed))
                context.stroke(axisPath, with: .style(.tertiary), lineWidth: 1)

                let minAge = ageBounds.min
                let maxAge = ageBounds.max
                let tickCount = 6
                for i in 0...tickCount {
                    let t = Double(i) / Double(tickCount)
                    let age = maxAge - (maxAge - minAge) * t
                    let logicalX = CGFloat(40 + (size.width - 60) * t)
                    let transformedPoint = transform(CGPoint(x: logicalX, y: axisYLogical))
                    let x = transformedPoint.x
                    let yAxis = transformedPoint.y

                    var tick = Path()
                    tick.move(to: CGPoint(x: x, y: yAxis - 4))
                    tick.addLine(to: CGPoint(x: x, y: yAxis + 4))
                    context.stroke(tick, with: .style(.tertiary), lineWidth: 1)

                    let label = Text("\(Int(age))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    context.draw(label, at: CGPoint(x: x, y: yAxis + 14), anchor: .center)
                }

                let titlePos = transform(CGPoint(x: 84, y: axisYLogical + 20))
                context.draw(Text("Возраст, лет назад").font(.caption).foregroundStyle(.secondary),
                             at: titlePos, anchor: .leading)

                for node in result.nodes {
                    guard let logicalPoint = logicalPositions[node.index] else { continue }
                    let point = transform(logicalPoint)
                    let isHighlighted = selectedNodeIndex == node.index
                    let isNeighbor = selectedNodeIndex != nil && result.networkEdges.contains(where: { edge in
                        (edge.a == selectedNodeIndex && edge.b == node.index) ||
                        (edge.b == selectedNodeIndex && edge.a == node.index)
                    }) && !isHighlighted

                    let radius: CGFloat = isHighlighted ? 14 : max(7, CGFloat(5 + sqrt(Double(node.count))))
                    let fillColor: Color = isHighlighted ? .red : (node.isObservedAge ? .blue : .orange)
                    let strokeColor: Color = isNeighbor ? .green : (isHighlighted ? .primary : .white)
                    let strokeWidth: CGFloat = isNeighbor ? 3 : 2

                    let rect = CGRect(x: point.x - radius, y: point.y - radius,
                                      width: radius * 2, height: radius * 2)
                    context.fill(Path(ellipseIn: rect), with: .color(fillColor))
                    context.stroke(Path(ellipseIn: rect), with: .color(strokeColor), lineWidth: strokeWidth)

                    let label = Text(node.label)
                        .font(.caption2)
                        .foregroundStyle(.primary)
                    context.draw(label, at: CGPoint(x: point.x, y: point.y - 16), anchor: .center)
                }
            }
            .background(Color(nsColor: .windowBackgroundColor))
            .onTapGesture { location in
                let logicalPositions = layout(size: canvasSize)
                let closest = findNode(at: location, logicalPositions: logicalPositions, size: canvasSize)
                if let idx = closest {
                    selectedNodeIndex = idx
                } else {
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if lastDragPosition == nil {
                            lastDragPosition = value.location
                        } else {
                            let deltaX = value.location.x - lastDragPosition!.x
                            let deltaY = value.location.y - lastDragPosition!.y
                            panOffset.width += deltaX
                            panOffset.height += deltaY
                            lastDragPosition = value.location
                        }
                    }
                    .onEnded { _ in
                        lastDragPosition = nil
                    }
            )
            .gesture(
                MagnificationGesture()
                    .onChanged { value in
                        let newZoom = zoomLevel * value
                        zoomLevel = min(max(newZoom, 0.3), 8.0)
                    }
            )
            .onAppear {
                eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
                    if event.modifierFlags.contains(.control) || event.modifierFlags.contains(.option) {
                        let delta = 1.0 + (event.deltaY > 0 ? 0.05 : -0.05)
                        let newZoom = zoomLevel * delta
                        zoomLevel = min(max(newZoom, 0.3), 8.0)
                        return nil
                    }
                    return event
                }
            }
            .onDisappear {
                if let monitor = eventMonitor {
                    NSEvent.removeMonitor(monitor)
                    eventMonitor = nil
                }
            }
            .onTapGesture(count: 2) {
                withAnimation(.spring()) {
                    zoomLevel = 1.0
                    panOffset = .zero
                }
            }
            .overlay(alignment: .topLeading) {
                if let idx = selectedNodeIndex, let node = result.nodes.first(where: { $0.index == idx }) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(node.label)
                            .font(.headline)
                        Text(node.isObservedAge ? "Датированный возраст: \(Int(node.observedAgeBP ?? 0)) лет" : "Прогнозный возраст: ~\(Int(node.inferredAgeBP ?? 0)) лет")
                        Text("Confidence: \(String(format: "%.3f", node.confidence))")
                        Text("Количество последовательностей: \(node.count)")
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Последовательности:")
                                .font(.caption.bold())
                            ForEach(node.memberHeaders.prefix(10), id: \.self) { seq in
                                Text(seq)
                                    .font(.caption2)
                            }
                            if node.memberHeaders.count > 10 {
                                Text("...")
                                    .font(.caption2)
                            }
                        }
                    }
                    .font(.caption)
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    .padding()
                }
            }
        }
    }

    private func layout(size: CGSize) -> [Int: CGPoint] {
        let nodes = result.nodes
        guard !nodes.isEmpty else { return [:] }

        let minAge = ageBounds.min
        let maxAge = ageBounds.max
        let width = max(size.width - 80, 200)
        let height = max(size.height - 70, 200)

        let count = max(nodes.count, 1)
        var positions: [Int: CGPoint] = [:]

        for (rank, node) in nodes.enumerated() {
            let age = node.inferredAgeBP ?? minAge
            let xT = (maxAge - age) / max(maxAge - minAge, 1e-9)
            let x = CGFloat(40 + width * xT)

            let baseY = CGFloat(30 + (height * Double(rank + 1) / Double(count + 1)))
            let hash = CGFloat(abs(node.label.hashValue % 97)) / 97.0
            let jitter = (hash - 0.5) * 70.0
            let y = min(max(baseY + jitter, 24), size.height - 44)
            positions[node.index] = CGPoint(x: x, y: y)
        }
        return positions
    }

    private func findNode(at location: CGPoint, logicalPositions: [Int: CGPoint], size: CGSize) -> Int? {
        func transform(_ point: CGPoint) -> CGPoint {
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let scaledX = center.x + (point.x - center.x) * zoomLevel + panOffset.width
            let scaledY = center.y + (point.y - center.y) * zoomLevel + panOffset.height
            return CGPoint(x: scaledX, y: scaledY)
        }

        var closestNode: Int? = nil
        var minDistance = CGFloat.greatestFiniteMagnitude

        for (idx, logicalPoint) in logicalPositions {
            let screenPoint = transform(logicalPoint)
            let radius: CGFloat = 12
            let distance = hypot(screenPoint.x - location.x, screenPoint.y - location.y)
            if distance <= radius && distance < minDistance {
                minDistance = distance
                closestNode = idx
            }
        }
        return closestNode
    }
}

// MARK: - Предпросмотр программы в Xcode до компиляции | Pre-compiled CANVAS (Xcode only)

#Preview {
    ContentView()
}
