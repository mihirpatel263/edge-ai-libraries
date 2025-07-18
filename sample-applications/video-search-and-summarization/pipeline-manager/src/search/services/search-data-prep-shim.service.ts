// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { HttpService } from '@nestjs/axios';
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { tap } from 'rxjs';
import { SearchEvents } from 'src/events/Pipeline.events';
import { DataPrepMinioDTO, DataPrepMinioRO } from '../model/search.model';

@Injectable()
export class SearchDataPrepShimService {
  constructor(
    private $config: ConfigService,
    private $http: HttpService,
    private $emitter: EventEmitter2,
  ) {}

  createEmbeddings(data: DataPrepMinioDTO) {
    const dataPrepEndpoint = this.$config.get('search.dataPrep')!;
    const api = [dataPrepEndpoint, 'videos', 'minio'].join('/');
    return this.$http.post<DataPrepMinioRO>(api, data).pipe(
      tap(() => {
        this.$emitter.emit(SearchEvents.EMBEDDINGS_UPDATE);
      }),
    );
  }
}
