# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None):
        """
        nl_inputs dediği doğal dil girdileri (comment vs.). onu kullanmıyoruz. o yüzden sadece ilk if tarafına girecek kod.
        position_idx dediği position vektörü. şöyle bir yapı kurulmuş, dfg node belirten <unk> tokenlarının hepsinin
        position değeri 0, <pad> tokenlarının position değeri 1, kalan code tokenları için ise sırasıyla 2,3,4,5...
        position idx ile kolaylıkla node maskesi isterse ya da token maskesi isterse alabiliyorlar, ilk 2 satırda olan o
        """
        if code_inputs is not None:

            nodes_mask=position_idx.eq(0)  # position vektörü ile nodeları getir
            token_mask=position_idx.ge(2)  # position vektörü ile tokenları getir

            # burada modeli direk çağırmak yerine, önce modelin word_embeddings katmanını kendisi çağırıyor.
            # sonra bunun çıktısı üzerinde işlemler yaptıktan sonra modelin encoder katmanını çağırıyorlar
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)

            # node maskesi & token maskesi & attention maskesi, dfg nodelarını tokenlara bağlayan maske
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask

            # her node için, birden fazla tokena attention'ı varsa toplamı 1 olacak şekilde bölüyor.
            # mesela bir dfg node'u 3 tane tokena bağlıysa, node'daki her değer 0.333 oluyor.
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]

            # bu katsayılar ile embedding katmanı çıktılarına, einstein summation notation kullanarak
            # abc, acd -> abd işlemini uyguluyorlar. bu işlem tam olarak ne bilmiyorum
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)

            # encoder'a verilecek son embeddingler şu şekilde belirleniyor:
            # node olmayan tokenlar (code tokenları ve padding tokenları) direk olarak embedding katmanından çıktığı gibi veriliyor.
            # node tokenları ise yukarıda hesaplanan sonuç olarak veriliyor.
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))

      
        
 
