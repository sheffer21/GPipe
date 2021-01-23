import os
import re
from os import listdir
from os.path import isfile, join, basename
import matplotlib.pyplot as plt
import pandas as pd

###################################################
# Consts
Latency_axis_title = "Total Latency [clock cycle]"
Latency_axis_title_per_message = "Msg. latency [clock cycle]"
Throughput_axis_title = "Throughput [message / clock cycle]"
Batch_size_axis_title = "Batch size [message]"
AveragePercentileName = 'average'
precentiles = [0.9, 0.99]
###################################################

###################################################
# Run data
def path2info(path):
    isConsumer = False
    pipe_type = None
    bufferSize = requestsCount = runsCount = threadCount = messageCount = syncRate = clockRate = 0
    for index, w in enumerate(re.split("[_.]", basename(path))):
        if w == 'naive':
            pipe_type = 'Naive pipe for single cpu process'
        if w == 'Three':
            pipe_type = 'Naive pipe'
        if w == 'Gpipe':
            pipe_type = 'GPipe' if pipe_type is None else pipe_type
        if w.startswith('QS'):
            bufferSize = int(w[2:])
        if w.startswith('Req'):
            requestsCount = int(w[3:])
        if w.startswith('Run'):
            runsCount = int(w[3:])
        if w.startswith('Thread'):
            threadCount = int(w[6:])
        if w.startswith('MS'):
            messageCount = int(w[2:])
        if w.startswith('SR'):
            syncRate = int(w[2:])
        if w.startswith('CR'):
            clockRate = int(w[2:])
        if w == 'consumer':
            isConsumer = True

    return bufferSize, requestsCount, runsCount, threadCount, messageCount, isConsumer, pipe_type, syncRate, clockRate


def get_message_count(data_latency):
    return [title2message_count(t) for t in data_latency.columns[::]]


def title2message_count(title):
    return int(re.findall(r"\[(\d+)\]", title)[0])

###################################################

###################################################
# Graphs

def getPrecentilesTitles(tiles):
    return [str(t) for t in tiles]


def find_percentiles(df, tiles, requestCount, dataOrderLowToHigh=True):
    precentiles_titles = getPrecentilesTitles(tiles)
    data_std = []
    data_mean = []
    precentiles_res = pd.DataFrame(columns=precentiles_titles)

    for index, (_, runResults) in enumerate(df.iteritems()):
        runResults_sorted = list(runResults.sort_values(ascending=dataOrderLowToHigh))
        percentiles_of_run = []

        for _, p in enumerate(tiles):
            row_idx = int(requestCount * p) - 1
            percentiles_of_run.append(runResults_sorted[row_idx])

        data_mean.append(runResults.mean())
        data_std.append(runResults.std())

        precentiles_res = precentiles_res.append(pd.DataFrame([percentiles_of_run], columns=precentiles_titles))

    return precentiles_res, data_std, data_mean


def calc_throughput(latency_dataFrame, message_counts):
    throughput_data = {}

    for index, [title, run] in enumerate(latency_dataFrame.iteritems()):
        current_message_count = message_counts[index]
        run_res = []
        for req_latency in run:
            run_res.append(int(current_message_count) / req_latency)

        throughput_data[title] = run_res

    return pd.DataFrame(throughput_data, columns=latency_dataFrame.columns)


def plot_percentile_graph(percentile_data, mean_data, std_data, ax, yLabel_title, message_counts, tiles, bufferSize):
    plotPrecentiles(tiles, percentile_data, message_counts, ax)
    plotMeanAndStd(message_counts, mean_data, std_data, ax)
    plotBufferSizeLine(bufferSize, ax)
    setMargins(percentile_data, mean_data, ax)
    ax.set_ylabel(yLabel_title)
    ax.legend()


def calc_average_latency(data_latency, message_counts):
    average_latency_data = {}

    for index, [title, run] in enumerate(data_latency.iteritems()):
        current_message_count = message_counts[index]
        run_res = []
        for req_latency in run:
            run_res.append(req_latency / int(current_message_count))

        average_latency_data[title] = run_res

    return pd.DataFrame(average_latency_data, columns=data_latency.columns)


def print_data_info(path):
    bufferSize, requestCount, runsCount, threadCount, messageCount, _, _, _, _ = path2info(path)
    print(f"Evaluate pipe from csv: \"{path}\"")
    print(f"    Buffer size: {bufferSize}[Number of messages]")
    print(f"    Number of requests per run: {requestCount}")
    print(f"    Number of runs: {runsCount}")
    print(f"    Number of Threads: {threadCount}")
    print(f"    Single message size: {messageCount}[Bytes]")

    message_counts = get_message_count(read_csv(path))
    print(f"    Message count in each run: {message_counts}")


def plot_data_results(csv_path):
    bufferSize, requestCount, _, _, _, isConsumer, pipeType, _, _ = path2info(csv_path)
    print_data_info(csv_path)

    data_latency = read_csv(csv_path)
    message_counts = get_message_count(data_latency)

    data_throughput = calc_throughput(data_latency, message_counts)
    data_average_latency = calc_average_latency(data_latency, message_counts)

    latency_precentiles_data, latency_data_std, latency_data_mean = find_percentiles(data_average_latency,
                                                                                     precentiles,
                                                                                     requestCount)
    throughput_precentiles_data, throughput_data_std, throughput_data_mean = find_percentiles(data_throughput,
                                                                                              precentiles,
                                                                                              requestCount,
                                                                                              dataOrderLowToHigh=False)

    fig, (latency_ax, throughput_ax) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex='col')
    plt.subplots_adjust(hspace=0.1)

    workerJob = 'Consumer' if isConsumer else 'Producer'
    fig.suptitle(f"{pipeType} - {workerJob}")

    plot_percentile_graph(
        latency_precentiles_data, latency_data_mean, latency_data_std, latency_ax,
        Latency_axis_title_per_message, message_counts, precentiles, bufferSize)

    plot_percentile_graph(
        throughput_precentiles_data, throughput_data_mean, throughput_data_std, throughput_ax,
        Throughput_axis_title, message_counts, precentiles, bufferSize)

    throughput_ax.set_xlabel(Batch_size_axis_title)
    plot_metadata(csv_path)


def plot_consumer_and_producer(producer, consumer):
    _, requestCount, _, _, _, _, pipeType, _, _ = path2info(consumer)

    consumer_latency = read_csv(consumer)
    producer_latency = read_csv(producer)

    messages_count = get_message_count(consumer_latency)

    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows=3, ncols=3)
    x = range(requestCount)
    axs = axs.flatten()
    for index, m_count in enumerate(messages_count):
        axs[index].set_title(f'Message count: {m_count}')
        axs[index].plot(x, producer_latency.iloc[:, index], label='producer')
        axs[index].plot(x, consumer_latency.iloc[:, index], label='Consumer')
        axs[index].legend()
        if index % 3 == 0:
            axs[index].set_ylabel(Latency_axis_title, fontsize=12)

        if index > 5:
            axs[index].set_xlabel('Repetition', fontsize=12)

        if index == 8:
            break

    plot_metadata(producer, fontSize=10, yPosition=0.9)
    fig.suptitle(f"{pipeType} - Latency vs Time", fontsize=14)


def compare_run_name(nameA, nameB):
    return remove_consumer_producer_from_name(nameA) == remove_consumer_producer_from_name(nameB)


def remove_consumer_producer_from_name(name):
    return name.replace('consumer', '').replace('producer', '')


def remove_pipe_name(name):
    return name.replace('Three_Element_Gpipe_', '').replace('Gpipe_', '').replace('naive_pipe_', '')


def plot_metadata(f, includeBufferSize=True, messageCount=None, includeSideType=False, fontSize=14, yPosition=0.86):
    bufferSize, requestCount, _, threadCount, messageSize, isConsumer, _, _, _ = path2info(f)

    plt.rcParams.update({'font.size': fontSize})
    bufferSizeStr = f"Buffer size: {bufferSize}[#messages]\n" if includeBufferSize else ""
    messageCountStr = f"Batch size: {messageCount}[#messages]\n" if messageCount is not None else ""
    pipeSideStr = f"Consumer side" if isConsumer else "Producer side"
    pipeSideStr = pipeSideStr if includeSideType else ""
    graph_metadata = \
        f"{bufferSizeStr}" \
        f"Message size: {messageSize}[Bytes]\n" \
        f"{messageCountStr}" \
        f"Repetitions: {requestCount}[#batches]\n" \
        f"#Threads: {threadCount}\n" \
        f"{pipeSideStr}"

    plt.figtext(0.72, yPosition, graph_metadata)


def plot_pipe_contest(pipeA, pipeB):
    bufferSize, requestCount, _, _, _, _, pipeTypeA, _, _ = path2info(pipeA)
    _, _, _, _, _, _, pipeTypeB, _, _ = path2info(pipeB)

    pipeA_latency = read_csv(pipeA)
    pipeB_latency = read_csv(pipeB)

    messages_count_A = get_message_count(pipeA_latency)
    messages_count_B = get_message_count(pipeB_latency)

    pipeA_throughput = calc_throughput(pipeA_latency, messages_count_A)
    pipeB_throughput = calc_throughput(pipeB_latency, messages_count_B)

    _, latency_data_std_A, latency_data_mean_A = find_percentiles(pipeA_latency, [], requestCount)
    _, latency_data_std_B, latency_data_mean_B = find_percentiles(pipeB_latency, [], requestCount)

    _, throughput_data_std_A, throughput_data_mean_A = find_percentiles(pipeA_throughput, [], requestCount, False)
    _, throughput_data_std_B, throughput_data_mean_B = find_percentiles(pipeB_throughput, [], requestCount, False)

    fig, (latency_ax, throughput_ax) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex='col')

    plot_metadata(pipeA, includeSideType=True, yPosition=0.88)

    # Plot pipes latency
    plotMeanAndStd(messages_count_A, latency_data_mean_A, latency_data_std_A, latency_ax, pipeTypeA)
    plotMeanAndStd(messages_count_B, latency_data_mean_B, latency_data_std_B, latency_ax, pipeTypeB)
    plotBufferSizeLine(bufferSize, latency_ax)
    latency_ax.set_ylabel(Latency_axis_title)
    latency_ax.legend()

    # Plot pipes throughput
    plotMeanAndStd(messages_count_A, throughput_data_mean_A, throughput_data_std_A, throughput_ax, pipeTypeA)
    plotMeanAndStd(messages_count_B, throughput_data_mean_B, throughput_data_std_B, throughput_ax, pipeTypeB)
    plotBufferSizeLine(bufferSize, throughput_ax)
    throughput_ax.set_ylabel(Throughput_axis_title)
    throughput_ax.legend()
    throughput_ax.set_xlabel(Batch_size_axis_title)

    fig.suptitle(f'{pipeTypeA} VS {pipeTypeB}')


def plotBufferSizeLine(bufferSize, ax=None):
    target = ax if ax is not None else plt
    target.axvline(bufferSize, label="Buffer size", color="red")


def compare_run_variables_with_no_pipe_type(runA, runB):
    run_A_no_type = remove_pipe_name(runA)
    run_B_no_type = remove_pipe_name(runB)

    if run_A_no_type != run_B_no_type:
        return False

    runA_latency = read_csv(runA)
    runB_latency = read_csv(runB)

    messages_count_A = get_message_count(runA_latency)
    messages_count_B = get_message_count(runB_latency)

    return messages_count_A == messages_count_B


def getResultFiles(dirName=""):
    results_dir_path = f'./{dirName}'
    files = [join(results_dir_path, f)
             for f in listdir(results_dir_path)
             if f.endswith('.csv') and isfile(join(results_dir_path, f))]

    files.sort(key=os.path.getmtime, reverse=True)
    return files


def plotMeanAndStd(x, mean, std, ax=None, label=None):
    target = ax if ax is not None else plt
    label = label if label is not None else AveragePercentileName
    target.scatter(x, mean)
    target.errorbar(x, mean, yerr=std, label=label)


def plotPrecentiles(tiles, percentilesLatency, x, ax=None):
    target = ax if ax is not None else plt
    tilesTitles = getPrecentilesTitles(tiles)
    for percentile in tilesTitles:
        data = percentilesLatency[percentile]
        target.scatter(x, data)
        target.plot(x, data, label=percentile)


def setMargins(precentilesLatency, meanLatency, ax=None, scope=0.15):
    max_y_value = max(precentilesLatency.max().max(), max(meanLatency))
    min_y_value = min(precentilesLatency.min().min(), min(meanLatency))
    margin = (max_y_value - min_y_value) * scope

    minYLim = min_y_value - margin
    maxYLim = max_y_value + margin
    target = ax.set_ylim if ax is not None else plt.ylim
    target(minYLim, maxYLim)


def plotParamVsLatency(files, getXValue, title, xLabel, paramName, includeBufferSizeInMetadata=True):
    _, requestCount, _, _, _, _, _, _, _ = path2info(files[0])

    files.sort(key=getXValue)
    xValueList = []
    averageLatency = []
    stdLatency = []
    percentilesLatency = pd.DataFrame(columns=getPrecentilesTitles(precentiles))

    for f in files:
        xValueList.append(getXValue(f))
        latency = read_csv(f)
        latency_precentiles, latency_std, latency_mean = find_percentiles(latency, precentiles, requestCount)

        averageLatency.append(latency_mean[0])
        stdLatency.append(latency_std[0])
        percentilesLatency = percentilesLatency.append(latency_precentiles)

    messageCount = get_message_count(read_csv(files[0]))[0]

    # Plot data
    plt.figure()
    plt.title(f"{title} : {paramName} VS Latency")
    plot_metadata(files[0], includeBufferSizeInMetadata, messageCount)

    # Add precentiles to graph
    plotPrecentiles(precentiles, percentilesLatency, xValueList)

    # Add average and std columns
    plotMeanAndStd(xValueList, averageLatency, stdLatency)
    setMargins(percentilesLatency, averageLatency)
    plt.legend()
    plt.ylabel(Latency_axis_title)
    plt.xlabel(xLabel)


def read_csv(f):
    return pd.read_csv(f).dropna(how='all', axis='columns')


def getPairs(files, comparator):
    return [(files[i], files[j])
            for i in range(0, len(files))
            for j in range(i, len(files))
            if files[i] != files[j] and
            comparator(files[i], files[j])]


def plotBufferSizeVSLatency():
    getBufferSizeValue = lambda elem: path2info(elem)[0]
    xTitle = "Buffer size [Messages]"
    paramName = "Buffer size"

    pipeType = ["gpipe", "threeElement"]
    pipeSide = ["consumer", "producer"]
    pipeSideTitle = {"gpipe": "GPipe", "threeElement": "Naive pipe"}

    for pType in pipeType:
        for side in pipeSide:
            bufferSizeTestFiles = getResultFiles(f"Size test/{pType}/{side}")
            plotParamVsLatency(bufferSizeTestFiles, getBufferSizeValue, f"{pipeSideTitle[pType]} {side}",
                               xTitle,
                               paramName,
                               includeBufferSizeInMetadata=False)


def plotSyncRateVSLatency():
    getSyncRateValue = lambda elem: path2info(elem)[7]
    xTitle = "Sync Rate [Integer per Thread]"
    paramName = "Sync Rate"

    pipeSide = ["consumer", "producer"]
    for side in pipeSide:
        syncRateTestFiles = getResultFiles(F"Sync rate optimization/Sync rate - small buffer/{side}")
        plotParamVsLatency(syncRateTestFiles, getSyncRateValue, f"GPipe {side}", xTitle, paramName)


def plotConsumerAndProducer():
    files = getResultFiles()
    result_pairs = getPairs(files, compare_run_name)
    result_pairs.reverse()

    for resultA, resultB in result_pairs:
        plot_consumer_and_producer(resultA, resultB)


def plotPipeContest():
    results_files = getResultFiles()
    pipes_pairs = getPairs(results_files, compare_run_variables_with_no_pipe_type)
    pipes_pairs.reverse()

    for resultA, resultB in pipes_pairs:
        plot_pipe_contest(resultA, resultB)


def plotLatencyThroughput():
    files = getResultFiles()
    for file in files:
        plot_data_results(file)


if __name__ == '__main__':
    plotBufferSizeVSLatency()
    plotSyncRateVSLatency()
    plotConsumerAndProducer()
    plotPipeContest()
    plotLatencyThroughput()

    plt.show()
