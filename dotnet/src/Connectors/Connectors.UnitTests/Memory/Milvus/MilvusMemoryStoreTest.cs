// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using IO.Milvus;
using IO.Milvus.Client;
using IO.Milvus.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.Connectors.Memory.Milvus;
using Microsoft.SemanticKernel.Memory;
using Moq;
using Xunit;

namespace SemanticKernel.Connectors.UnitTests.Memory.Milvus;

public class MilvusMemoryStoreTests
{
    private readonly string _id = "Id";
    private readonly string _id2 = "Id2";
    private readonly string _id3 = "Id3";

    private readonly string _text = "text";
    private readonly string _text2 = "text2";
    private readonly string _text3 = "text3";

    private readonly string _description = "description";
    private readonly string _description2 = "description2";
    private readonly string _description3 = "description3";

    private readonly Embedding<float> _embedding = new(new float[] { 1, 1, 1 });
    private readonly Embedding<float> _embedding2 = new(new float[] { 2, 2, 2 });
    private readonly Embedding<float> _embedding3 = new(new float[] { 3, 3, 3 });

    private readonly Mock<IMilvusClient> _mockMilvusClient;
    private readonly Mock<ILogger<IMilvusMemoryStore>> _mockLogger = new();

    public MilvusMemoryStoreTests()
    {
        this._mockMilvusClient = new Mock<IMilvusClient>();
    }

    [Fact]
    public void ConnectionCanBeInitialized()
    {
        // Arrange & Act
        using MilvusMemoryStore memoryStore = new(
            this._mockMilvusClient.Object,
            1536,
            log: this._mockLogger.Object);

        // Assert
        Assert.NotNull(memoryStore);
    }

    [Fact]
    public async Task ItCreatesNewCollectionAsync()
    {
        string collectionName = "test";

        // Arrange
        this._mockMilvusClient
            .Setup(x => x.HasCollectionAsync(
                It.IsAny<string>(),
                It.IsAny<DateTime?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(false);

        using var vectorStore = new MilvusMemoryStore(
            this._mockMilvusClient.Object,
            1536,
            log: this._mockLogger.Object);

        // Act
        await vectorStore.CreateCollectionAsync(collectionName);

        // Assert
        this._mockMilvusClient
            .Verify<Task<bool>>(x => x.HasCollectionAsync(
                collectionName,
                It.IsAny<DateTime?>(),
                It.IsAny<CancellationToken>()),
                Times.Once());
        this._mockMilvusClient
            .Verify<Task>(x => x.CreateCollectionAsync(
                collectionName,
                It.IsAny<IList<FieldType>>(),
                It.IsAny<MilvusConsistencyLevel>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()),
                Times.Once);
        this._mockMilvusClient
            .Verify<Task>(x => x.CreateIndexAsync(
                collectionName,
                "embedding",
                Constants.DEFAULT_INDEX_NAME,
                MilvusIndexType.AUTOINDEX,
                MilvusMetricType.IP,
                null,
                It.IsAny<CancellationToken>()), Times.Once);
        this._mockMilvusClient
            .Verify<Task>(x => x.LoadCollectionAsync(
                collectionName,
                1,
                It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ItWillNotOverwriteExistingCollectionAsync()
    {
        string collectionName = "test";

        // Arrange
        this._mockMilvusClient
           .Setup(x => x.HasCollectionAsync(
               It.IsAny<string>(),
               It.IsAny<DateTime?>(),
               It.IsAny<CancellationToken>()))
           .ReturnsAsync(true);

        using var vectorStore = new MilvusMemoryStore(
            this._mockMilvusClient.Object,
            1536,
            log: this._mockLogger.Object);

        // Act
        await vectorStore.CreateCollectionAsync(collectionName);

        // Assert
        this._mockMilvusClient
           .Verify<Task<bool>>(x => x.HasCollectionAsync(
               collectionName,
               It.IsAny<DateTime?>(),
               It.IsAny<CancellationToken>()),
               Times.Once());
        this._mockMilvusClient
            .Verify<Task>(x => x.CreateCollectionAsync(
                collectionName,
                It.IsAny<IList<FieldType>>(),
                It.IsAny<MilvusConsistencyLevel>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()),
                Times.Never);
    }

    [Fact]
    public async Task ItDeleteCollectionAsync()
    {
        string collectionName = "test";

        // Arrange
        this._mockMilvusClient
           .Setup(x => x.HasCollectionAsync(
               It.IsAny<string>(),
               It.IsAny<DateTime?>(),
               It.IsAny<CancellationToken>()))
           .ReturnsAsync(true);
        this._mockMilvusClient
           .Setup(x => x.DropCollectionAsync(
               It.IsAny<string>(),
               It.IsAny<CancellationToken>()));

        using var vectorStore = new MilvusMemoryStore(
            this._mockMilvusClient.Object,
            1536,
            log: this._mockLogger.Object);

        // Act
        await vectorStore.DeleteCollectionAsync(collectionName);

        // Assert
        this._mockMilvusClient
           .Verify(x => x.DropCollectionAsync(
               collectionName,
               It.IsAny<CancellationToken>()),
               Times.Once);
    }

    [Fact]
    public async Task ItThrowsIfUpsertRequestFailAsync()
    {
        string collectionName = "test_collection";

        // Arrange
        var memoryRecord = MemoryRecord.LocalRecord(
            id: this._id,
            text: this._text,
            description: this._description,
            embedding: this._embedding);

        this._mockMilvusClient
            .Setup(x => x.InsertAsync(
               It.IsAny<string>(),
               It.IsAny<IList<Field>>(),
               It.IsAny<string>(),
               It.IsAny<CancellationToken>()))
            .Throws<MilvusException>();

        using var vectorStore = new MilvusMemoryStore(
            this._mockMilvusClient.Object,
            1536,
            log: this._mockLogger.Object);

        // Assert
        await Assert.ThrowsAsync<MilvusException>(() => vectorStore.UpsertAsync(collectionName, memoryRecord));
    }

    [Fact]
    public async Task InsertIntoNonExistentCollectionDoesNotCallCreateCollectionAsync()
    {

    }

    [Fact]
    public async Task ItUpdatesExistingDataEntryBasedOnMetadataIdAsync()
    {

    }

    [Fact]
    public async Task ItCanBatchUpsertAsync()
    {

    }
}
